"""
ml_model.py
Machine Learning: kNN and Naive Bayes for victim survival probability
and area risk severity estimation.
Uses NumPy for fast vectorised array operations and distance calculations.
"""

from typing import List, Tuple, Dict
import numpy as np


# ── Synthetic Dataset ──────────────────────────────────────────────────────────

def generate_dataset(n: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Features per victim:
      [severity_score (1-3), distance_to_center, risk_zone_nearby (0/1),
       time_elapsed, medical_kits_available]
    Label: 1 = survives, 0 = does not survive
    Returns NumPy arrays for fast vectorised ops.
    """
    rng = np.random.default_rng(seed)

    severity     = rng.integers(1, 4, size=n)               # 1, 2, or 3
    distance     = rng.uniform(1, 15, size=n)
    risk_nearby  = rng.integers(0, 2, size=n)               # 0 or 1
    time_elapsed = rng.uniform(0, 30, size=n)
    kits         = rng.integers(0, 11, size=n)

    score = (
        (4 - severity) * 2.0
        - distance * 0.3
        - risk_nearby * 1.5
        - time_elapsed * 0.1
        + kits * 0.2
        + rng.normal(0, 0.5, size=n)
    )
    labels = (score > 2.5).astype(int)

    X = np.column_stack([severity, distance, risk_nearby, time_elapsed, kits]).astype(float)
    return X, labels


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy  = (tp + tn) / len(y_true) if len(y_true) else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def print_metrics(metrics: Dict, label: str):
    print(f"\n[ML — {label}]")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1-Score : {metrics['f1']:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")


# ── k-Nearest Neighbours (NumPy vectorised) ────────────────────────────────────

class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Euclidean distance from x to all training points."""
        diff = self.X_train - x          # (n_train, n_features)
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def predict_one(self, x) -> int:
        x = np.asarray(x, dtype=float)
        dists = self._distances(x)
        nn_idx = np.argpartition(dists, self.k)[:self.k]
        votes = np.bincount(self.y_train[nn_idx].astype(int))
        return int(np.argmax(votes))

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X])

    def predict_proba(self, x) -> float:
        """Returns survival probability (fraction of positive neighbours)."""
        x = np.asarray(x, dtype=float)
        dists = self._distances(x)
        nn_idx = np.argpartition(dists, self.k)[:self.k]
        return float(np.mean(self.y_train[nn_idx]))


# ── Naive Bayes (Gaussian, NumPy) ─────────────────────────────────────────────

class GaussianNaiveBayes:
    def __init__(self):
        self.classes_:       np.ndarray = np.array([])
        self.class_priors_:  np.ndarray = np.array([])
        self.means_:         np.ndarray = np.array([])   # (n_classes, n_features)
        self.vars_:          np.ndarray = np.array([])   # (n_classes, n_features)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.vars_  = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes_])

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_classes) log-likelihoods."""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes_)))
        for i, (mean, var) in enumerate(zip(self.means_, self.vars_)):
            log_prior = np.log(self.class_priors_[i])
            log_lh    = -0.5 * np.sum(np.log(2 * np.pi * var)
                                       + (X - mean) ** 2 / var, axis=1)
            log_probs[:, i] = log_prior + log_lh
        return log_probs

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_probs = self._log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_one(self, x) -> int:
        return int(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def predict_proba(self, x) -> float:
        """Returns survival probability (posterior for class=1)."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        log_probs = self._log_likelihood(x)[0]
        # Numerically stable softmax
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        class1_idx = np.where(self.classes_ == 1)[0]
        return float(probs[class1_idx[0]]) if len(class1_idx) else 0.0


# ── Training pipeline ──────────────────────────────────────────────────────────

def train_and_evaluate():
    """Train both models, print metrics, return trained models."""
    X, y = generate_dataset(n=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    knn_preds   = knn.predict(X_test)
    knn_metrics = compute_metrics(y_test, knn_preds)
    print_metrics(knn_metrics, "kNN (k=5)")

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    nb_preds   = nb.predict(X_test)
    nb_metrics = compute_metrics(y_test, nb_preds)
    print_metrics(nb_metrics, "Gaussian Naive Bayes")

    return knn, nb, knn_metrics, nb_metrics


# ── Victim feature extractor ───────────────────────────────────────────────────

def victim_features(victim, distance: float,
                    risk_nearby: int, time_elapsed: float,
                    kits_remaining: int) -> np.ndarray:
    return np.array([
        float(victim.priority),
        distance,
        float(risk_nearby),
        time_elapsed,
        float(kits_remaining),
    ])