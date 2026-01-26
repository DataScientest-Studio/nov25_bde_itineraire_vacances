# src/tsp/benchmark/metrics.py

import numpy as np

def basic_stats(costs, times):
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "mean_time": float(np.mean(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
    }

def stability_metrics(costs):
    mean = np.mean(costs)
    std = np.std(costs)
    cv = std / mean if mean > 0 else 0
    return {
        "std_cost": float(std),
        "cv_cost": float(cv),
        "robustness": float(1 / cv) if cv > 0 else float("inf"),
    }

def quality_metrics(costs, global_best):
    mean_cost = np.mean(costs)
    gap = (mean_cost - global_best) / global_best
    eff = global_best / mean_cost
    return {
        "gap_vs_best": float(gap),
        "efficiency": float(eff),
    }

def structural_metrics(route, D):
    edges = np.array([D[route[i], route[i+1]] for i in range(len(route)-1)])
    return {
        "max_edge": float(edges.max()),
        "mean_edge": float(edges.mean()),
        "edge_ratio": float(edges.max() / edges.mean()),
        "long_jumps": int(np.sum(edges > edges.mean() + 2 * edges.std())),
    }

def pareto_score(mean_cost, mean_time, max_cost, max_time, alpha=0.7):
    return float(alpha * (mean_cost / max_cost) + (1 - alpha) * (mean_time / max_time))

def tsp_rating(efficiency, robustness, pareto):
    return float(0.4 * efficiency + 0.3 * robustness + 0.3 * (1 - pareto))