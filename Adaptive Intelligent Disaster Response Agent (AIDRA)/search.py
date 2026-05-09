"""
search.py
Search algorithms: BFS, DFS, Greedy Best-First, A*, Hill Climbing.
All return (path, nodes_expanded, cost) for KPI reporting.
"""

import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict, Callable
from environment import GridMap, CELL_HIGH_RISK


# ── Cost helpers ──────────────────────────────────────────────────────────────

RISK_PENALTY = 5   # extra cost per high-risk cell traversed

def step_cost(grid_map: GridMap, pos: Tuple[int,int], avoid_risk: bool = False) -> float:
    base = 1.0
    if grid_map.is_high_risk(pos):
        return base + (RISK_PENALTY if avoid_risk else 0)
    return base

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ── Result dataclass ───────────────────────────────────────────────────────────

class SearchResult:
    def __init__(self, algorithm: str, path: List[Tuple[int,int]],
                 nodes_expanded: int, cost: float, risk_cells: int):
        self.algorithm     = algorithm
        self.path          = path
        self.nodes_expanded = nodes_expanded
        self.cost          = cost
        self.risk_cells    = risk_cells  # number of high-risk cells on path
        self.found         = len(path) > 0

    def summary(self) -> str:
        if not self.found:
            return f"[{self.algorithm}] No path found. Nodes expanded: {self.nodes_expanded}"
        return (f"[{self.algorithm}] Path length={len(self.path)-1} steps | "
                f"Cost={self.cost:.1f} | Risk cells={self.risk_cells} | "
                f"Nodes expanded={self.nodes_expanded}")


def _reconstruct(came_from: Dict, start, goal) -> List[Tuple[int,int]]:
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    if path and path[0] == start:
        return path
    return []


def _count_risk(grid_map: GridMap, path: List[Tuple[int,int]]) -> int:
    return sum(1 for p in path if grid_map.is_high_risk(p))


# ── BFS ───────────────────────────────────────────────────────────────────────

def bfs(grid_map: GridMap, start: Tuple[int,int],
        goal: Tuple[int,int]) -> SearchResult:
    frontier   = deque([start])
    came_from  = {start: None}
    expanded   = 0

    while frontier:
        node = frontier.popleft()
        expanded += 1
        if node == goal:
            path = _reconstruct(came_from, start, goal)
            return SearchResult("BFS", path, expanded, len(path)-1, _count_risk(grid_map, path))
        for nb in grid_map.neighbors(node):
            if nb not in came_from:
                came_from[nb] = node
                frontier.append(nb)

    return SearchResult("BFS", [], expanded, float('inf'), 0)


# ── DFS ───────────────────────────────────────────────────────────────────────

def dfs(grid_map: GridMap, start: Tuple[int,int],
        goal: Tuple[int,int], depth_limit: int = 200) -> SearchResult:
    stack     = [(start, [start])]
    visited   = set()
    expanded  = 0

    while stack:
        node, path = stack.pop()
        if node in visited or len(path) > depth_limit:
            continue
        visited.add(node)
        expanded += 1
        if node == goal:
            return SearchResult("DFS", path, expanded, len(path)-1, _count_risk(grid_map, path))
        for nb in grid_map.neighbors(node):
            if nb not in visited:
                stack.append((nb, path + [nb]))

    return SearchResult("DFS", [], expanded, float('inf'), 0)


# ── Greedy Best-First ─────────────────────────────────────────────────────────

def greedy_best_first(grid_map: GridMap, start: Tuple[int,int],
                      goal: Tuple[int,int]) -> SearchResult:
    heap      = [(manhattan(start, goal), start)]
    came_from = {start: None}
    expanded  = 0

    while heap:
        _, node = heapq.heappop(heap)
        expanded += 1
        if node == goal:
            path = _reconstruct(came_from, start, goal)
            return SearchResult("Greedy", path, expanded, len(path)-1, _count_risk(grid_map, path))
        for nb in grid_map.neighbors(node):
            if nb not in came_from:
                came_from[nb] = node
                heapq.heappush(heap, (manhattan(nb, goal), nb))

    return SearchResult("Greedy", [], expanded, float('inf'), 0)


# ── A* ────────────────────────────────────────────────────────────────────────

def astar(grid_map: GridMap, start: Tuple[int,int],
          goal: Tuple[int,int], avoid_risk: bool = False) -> SearchResult:
    """
    A* with optional risk-penalised cost.
    avoid_risk=True → adds RISK_PENALTY to high-risk cells (safer route).
    avoid_risk=False → purely shortest path.
    """
    g_cost    = {start: 0.0}
    came_from = {start: None}
    heap      = [(manhattan(start, goal), 0.0, start)]
    expanded  = 0

    while heap:
        f, g, node = heapq.heappop(heap)
        if g > g_cost.get(node, float('inf')):
            continue   # stale entry
        expanded += 1
        if node == goal:
            path = _reconstruct(came_from, start, goal)
            return SearchResult("A*", path, expanded, g, _count_risk(grid_map, path))
        for nb in grid_map.neighbors(node):
            new_g = g + step_cost(grid_map, nb, avoid_risk)
            if new_g < g_cost.get(nb, float('inf')):
                g_cost[nb]    = new_g
                came_from[nb] = node
                h             = manhattan(nb, goal)
                heapq.heappush(heap, (new_g + h, new_g, nb))

    return SearchResult("A*", [], expanded, float('inf'), 0)


# ── Hill Climbing (local search for route optimisation) ───────────────────────

def hill_climbing(grid_map: GridMap, start: Tuple[int,int],
                  goal: Tuple[int,int], max_restarts: int = 5) -> SearchResult:
    """
    Steepest-ascent hill climbing with random restarts.
    Heuristic: negative manhattan distance (maximise = minimise distance).
    """
    best_path: List[Tuple[int,int]] = []
    best_cost = float('inf')
    total_expanded = 0

    for _ in range(max_restarts):
        current = start
        path    = [current]
        visited = {current}
        expanded = 0
        stuck   = False

        while current != goal:
            neighbours = [nb for nb in grid_map.neighbors(current) if nb not in visited]
            if not neighbours:
                stuck = True
                break
            # pick neighbour closest to goal
            next_node = min(neighbours, key=lambda nb: manhattan(nb, goal))
            if manhattan(next_node, goal) >= manhattan(current, goal):
                stuck = True
                break   # local minimum
            current = next_node
            path.append(current)
            visited.add(current)
            expanded += 1

        total_expanded += expanded
        if not stuck and current == goal:
            cost = len(path) - 1
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]

    if best_path:
        return SearchResult("HillClimbing", best_path, total_expanded,
                            best_cost, _count_risk(grid_map, best_path))
    return SearchResult("HillClimbing", [], total_expanded, float('inf'), 0)


# ── Route Selector ────────────────────────────────────────────────────────────

def find_best_route(grid_map: GridMap, start: Tuple[int,int],
                    goal: Tuple[int,int], strategy: str = "astar_safe") -> SearchResult:
    """
    Unified interface used by the agent. strategy choices:
      'bfs', 'dfs', 'greedy', 'astar_fast', 'astar_safe', 'hill_climbing'
    """
    if strategy == "bfs":
        return bfs(grid_map, start, goal)
    elif strategy == "dfs":
        return dfs(grid_map, start, goal)
    elif strategy == "greedy":
        return greedy_best_first(grid_map, start, goal)
    elif strategy == "astar_fast":
        return astar(grid_map, start, goal, avoid_risk=False)
    elif strategy == "astar_safe":
        return astar(grid_map, start, goal, avoid_risk=True)
    elif strategy == "hill_climbing":
        return hill_climbing(grid_map, start, goal)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Compare all algorithms ────────────────────────────────────────────────────

def compare_algorithms(grid_map: GridMap,
                       start: Tuple[int,int],
                       goal: Tuple[int,int]) -> List[SearchResult]:
    results = [
        bfs(grid_map, start, goal),
        dfs(grid_map, start, goal),
        greedy_best_first(grid_map, start, goal),
        astar(grid_map, start, goal, avoid_risk=False),
        astar(grid_map, start, goal, avoid_risk=True),
        hill_climbing(grid_map, start, goal),
    ]
    # rename last A* for clarity
    results[4].algorithm = "A*(safe)"
    results[3].algorithm = "A*(fast)"
    return results