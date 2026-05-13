"""
ml_model.py
KTAS training utilities for AIDRA.

This module trains lightweight NumPy-based KNN and Gaussian Naive Bayes
models on the emergency triage dataset and keeps a small compatibility
layer for the simulator.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_FEATURE_COLUMNS = ["Age", "Injury", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT"]
TARGET_COLUMN_CANDIDATES = ["KTAS_expert", "KTAS_RN", "KTAS"]
KTAS_TO_CATEGORY = {1: "critical", 2: "critical", 3: "moderate", 4: "moderate", 5: "minor"}
KTAS_URGENCY_WEIGHTS = {1: 1.00, 2: 0.80, 3: 0.55, 4: 0.30, 5: 0.00}


def _normalise_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


_HEADER_ALIASES = {
    "group": "Group",
    "sex": "Sex",
    "age": "Age",
    "patientsnumberperhour": "Patients number per hour",
    "arrivalmode": "Arrival mode",
    "injury": "Injury",
    "chiefcomplain": "Chief_complain",
    "mental": "Mental",
    "pain": "Pain",
    "nrspain": "NRS_pain",
    "sbp": "SBP",
    "dbp": "DBP",
    "hr": "HR",
    "rr": "RR",
    "bt": "BT",
    "saturation": "Saturation",
    "ktasrn": "KTAS_RN",
    "ktas": "KTAS",
    "diagnosisined": "Diagnosis in ED",
    "disposition": "Disposition",
    "ktasexpert": "KTAS_expert",
    "errorgroup": "Error_group",
    "lengthofstaymin": "Length of stay_min",
    "ktasdurationmin": "KTAS duration_min",
    "mistriage": "mistriage",
}


def _canonical_name(name: str) -> str:
    key = _normalise_name(name.replace(",", ""))
    return _HEADER_ALIASES.get(key, name.strip().strip(","))


def _to_float(value) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        return float("nan")

    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _to_binary_injury(value) -> float:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "nan"}:
        return float("nan")
    if text in {"1", "0", "no", "false", "n"}:
        return 0.0
    if text in {"2", "yes", "true", "y"}:
        return 1.0
    try:
        return 1.0 if int(float(text)) >= 2 else 0.0
    except ValueError:
        return float("nan")


def _to_target(value) -> Optional[int]:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    text = text.replace(",", ".")
    try:
        target = int(float(text))
    except ValueError:
        return None
    return target if 1 <= target <= 5 else None


def _fill_missing_columnwise(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.astype(float, copy=True)
    for col in range(filled.shape[1]):
        column = filled[:, col]
        mask = np.isnan(column)
        if np.all(mask):
            filled[:, col] = 0.0
            continue
        median = float(np.nanmedian(column))
        if np.isnan(median):
            median = 0.0
        filled[mask, col] = median
    return filled


def _resolve_dataset_path(dataset_path: Optional[str | Path]) -> Path:
    candidates: List[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path))
    candidates.extend([Path("dataset_clean.csv"), Path("dataset.csv"), Path("data.csv")])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No cleaned dataset found. Place dataset_clean.csv next to ml_model.py.")


def _read_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(handle, dialect)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"Empty dataset file: {path}")

    raw_header = [part.strip() for part in rows[0]]
    header = [_canonical_name(part) for part in raw_header]
    expected_cols = len(header)

    repaired_rows: List[List[str]] = []
    for row in rows[1:]:
        parts = [part.strip() for part in row]
        if len(parts) > expected_cols:
            parts = parts[: expected_cols - 1] + [",".join(parts[expected_cols - 1:]).strip()]
        elif len(parts) < expected_cols:
            parts.extend([""] * (expected_cols - len(parts)))
        repaired_rows.append(parts[:expected_cols])
    return header, repaired_rows


def load_triage_dataset(
    dataset_path: Optional[str | Path] = None,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the triage dataset and return (X, y, feature_columns_used).

    The loader keeps row boundaries intact, preserves missing values as NaN,
    and only imputes the selected training columns.
    """

    path = _resolve_dataset_path(dataset_path)
    header, rows = _read_rows(path)

    header_index = {_normalise_name(col): idx for idx, col in enumerate(header)}

    resolved_features: List[str] = []
    feature_indices: List[int] = []
    for col in feature_columns:
        key = _normalise_name(col)
        if key not in header_index:
            raise KeyError(f"Feature column '{col}' not found in {path.name}")
        resolved = header[header_index[key]]
        resolved_features.append(resolved)
        feature_indices.append(header_index[key])

    if target_column is None:
        for candidate in TARGET_COLUMN_CANDIDATES:
            key = _normalise_name(candidate)
            if key in header_index:
                target_column = header[header_index[key]]
                break
    if target_column is None:
        raise KeyError(
            f"No KTAS target column found in {path.name}. "
            f"Expected one of: {', '.join(TARGET_COLUMN_CANDIDATES)}"
        )
    target_index = header_index[_normalise_name(target_column)]

    feature_rows: List[List[float]] = []
    targets: List[int] = []
    for row in rows:
        target = _to_target(row[target_index])
        if target is None:
            continue

        feature_vector: List[float] = []
        for idx, col_name in zip(feature_indices, resolved_features):
            value = row[idx]
            if _normalise_name(col_name) == "injury":
                feature_vector.append(_to_binary_injury(value))
            else:
                feature_vector.append(_to_float(value))
        feature_rows.append(feature_vector)
        targets.append(target)

    if not feature_rows:
        raise ValueError(f"No usable rows found in {path.name}")

    X = _fill_missing_columnwise(np.asarray(feature_rows, dtype=float))
    y = np.asarray(targets, dtype=int)
    return X, y, resolved_features


class FeatureScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler has not been fitted.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = max(1, int(len(X) * (1 - test_ratio)))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _class_probability_vector(classes: np.ndarray, probs: np.ndarray) -> np.ndarray:
    full = np.zeros(5, dtype=float)
    for cls, prob in zip(classes, probs):
        if 1 <= int(cls) <= 5:
            full[int(cls) - 1] = float(prob)
    total = full.sum()
    if total > 0:
        full /= total
    return full


def _urgency_score_from_probs(classes: np.ndarray, probs: np.ndarray) -> float:
    class_probs = _class_probability_vector(classes, probs)
    score = 0.0
    for ktas, prob in enumerate(class_probs, start=1):
        score += KTAS_URGENCY_WEIGHTS[ktas] * prob
    return float(score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.array(sorted(set(y_true.tolist()) | set(y_pred.tolist())), dtype=int)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        matrix[class_to_idx[int(yt)], class_to_idx[int(yp)]] += 1

    confusion = {}
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        tn = int(len(y_true) - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confusion[int(cls)] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "classes": classes.tolist(),
        "confusion_matrix": matrix.tolist(),
        "confusion": confusion,
    }


def print_metrics(metrics: Dict[str, object], label: str):
    print(f"\n[ML - {label}]")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1-Score : {metrics['f1']:.3f}")
    print("  Confusion summary:")
    for cls in metrics.get("classes", []):
        row = metrics["confusion"][cls]
        print(
            f"    KTAS {cls}: TP={row['tp']} FP={row['fp']} "
            f"FN={row['fn']} TN={row['tn']}"
        )


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.scaler: Optional[FeatureScaler] = None

    def fit(self, X, y, scaler: Optional[FeatureScaler] = None):
        self.scaler = scaler
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        self.X_train = X
        self.y_train = np.asarray(y, dtype=int)
        return self

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.scaler is not None:
            x = self.scaler.transform(np.atleast_2d(x))[0]
        return x

    def _distances(self, x: np.ndarray) -> np.ndarray:
        diff = self.X_train - x
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _neighbor_probs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self._prepare(x)
        dists = self._distances(x)
        k = min(self.k, len(self.y_train))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        labels = self.y_train[nn_idx]
        classes = np.array(sorted(set(self.y_train.tolist())), dtype=int)
        probs = np.array([np.mean(labels == cls) for cls in classes], dtype=float)
        return classes, probs

    def predict_one(self, x) -> int:
        classes, probs = self._neighbor_probs(x)
        return int(classes[int(np.argmax(probs))])

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X], dtype=int)

    def predict_proba(self, x) -> float:
        classes, probs = self._neighbor_probs(x)
        return _urgency_score_from_probs(classes, probs)


class GaussianNaiveBayes:
    def __init__(self):
        self.classes_: np.ndarray = np.array([])
        self.class_priors_: np.ndarray = np.array([])
        self.means_: np.ndarray = np.array([])
        self.vars_: np.ndarray = np.array([])
        self.scaler: Optional[FeatureScaler] = None

    def fit(self, X, y, scaler: Optional[FeatureScaler] = None):
        self.scaler = scaler
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.array(sorted(np.unique(y).tolist()), dtype=int)
        self.class_priors_ = np.array([np.mean(y == c) for c in self.classes_], dtype=float)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_], dtype=float)
        self.vars_ = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes_], dtype=float)
        return self

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare(X)
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes_)), dtype=float)
        for i, (mean, var) in enumerate(zip(self.means_, self.vars_)):
            log_prior = np.log(self.class_priors_[i] + 1e-12)
            log_lh = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
            log_probs[:, i] = log_prior + log_lh
        return log_probs

    def _class_probs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        log_probs = self._log_likelihood(x)[0]
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        return self.classes_, probs

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_probs = self._log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)].astype(int)

    def predict_one(self, x) -> int:
        classes, probs = self._class_probs(np.asarray(x, dtype=float))
        return int(classes[int(np.argmax(probs))])

    def predict_proba(self, x) -> float:
        classes, probs = self._class_probs(np.asarray(x, dtype=float))
        return _urgency_score_from_probs(classes, probs)


def train_and_evaluate(
    dataset_path: Optional[str | Path] = None,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Train both models on the triage dataset and return them plus metrics."""

    X, y, used_features = load_triage_dataset(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=test_ratio, seed=seed)

    scaler = FeatureScaler().fit(X_train)

    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train, scaler=scaler)
    knn_preds = knn.predict(X_test)
    knn_metrics = compute_metrics(y_test, knn_preds)
    print_metrics(knn_metrics, "kNN (KTAS)")

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train, scaler=scaler)
    nb_preds = nb.predict(X_test)
    nb_metrics = compute_metrics(y_test, nb_preds)
    print_metrics(nb_metrics, "Gaussian Naive Bayes (KTAS)")

    print(f"\n[ML] Training dataset: {dataset_path or 'auto-detected'}")
    print(f"[ML] Features used: {', '.join(used_features)}")
    print(f"[ML] Samples used: {len(X)}")

    return knn, nb, knn_metrics, nb_metrics


def build_proxy_triage_features(
    victim,
    distance: float,
    risk_nearby: int,
    time_elapsed: float,
    kits_remaining: int,
) -> np.ndarray:
    """
    Build a proxy triage feature vector for the simulator.

    The simulator does not store real patient vitals, so this helper creates a
    stable, severity-aware estimate that can still exercise the KTAS model.
    """

    priority = int(getattr(victim, "priority", 2))
    priority = max(1, min(priority, 3))

    age_map = {3: 72.0, 2: 54.0, 1: 31.0}
    injury_map = {3: 1.0, 2: 1.0, 1: 0.0}
    pain_map = {3: 8.5, 2: 5.5, 1: 2.0}
    sbp_map = {3: 92.0, 2: 110.0, 1: 124.0}
    dbp_map = {3: 58.0, 2: 70.0, 1: 80.0}
    hr_map = {3: 118.0, 2: 98.0, 1: 82.0}
    rr_map = {3: 26.0, 2: 22.0, 1: 18.0}
    bt_map = {3: 38.0, 2: 37.4, 1: 36.8}

    age = age_map[priority] + min(distance, 12.0) * 0.15
    injury = injury_map[priority]
    nrs_pain = pain_map[priority] + (0.5 if risk_nearby else 0.0)
    sbp = sbp_map[priority] - min(time_elapsed, 20.0) * 0.2
    dbp = dbp_map[priority]
    hr = hr_map[priority] + (3.0 if risk_nearby else 0.0)
    rr = rr_map[priority] + (1.0 if kits_remaining == 0 else 0.0)
    bt = bt_map[priority]

    return np.array([age, injury, nrs_pain, sbp, dbp, hr, rr, bt], dtype=float)


def victim_features(
    victim,
    distance: float,
    risk_nearby: int,
    time_elapsed: float,
    kits_remaining: int,
) -> np.ndarray:
    return build_proxy_triage_features(victim, distance, risk_nearby, time_elapsed, kits_remaining)


def map_ktas_to_category(ktas: int) -> str:
    return KTAS_TO_CATEGORY.get(int(ktas), "moderate")


if __name__ == "__main__":
    train_and_evaluate()
