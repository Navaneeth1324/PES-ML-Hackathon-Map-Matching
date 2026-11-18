"""
Evaluation & Benchmark Suite
============================
Computes quantitative evaluation metrics for map matching:
- Arc Precision, Recall, and F1 Score
- Longest Common Subsequence (LCS) & Sequence Accuracy
- Topological Continuity Check (valid graph path verification)
- End-to-end Latency (seconds)
"""

def evaluate_route(pred_route, gt_route, arcs=None):
    """
    Evaluates a predicted route against ground truth.
    """
    pred_set = set(pred_route)
    gt_set = set(gt_route)
    common = pred_set.intersection(gt_set)

    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Longest Common Subsequence (LCS)
    n, m = len(pred_route), len(gt_route)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if pred_route[i - 1] == gt_route[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    lcs = dp[m]
    lcs_accuracy = lcs / max(n, m) if max(n, m) > 0 else 0.0

    # Topological Continuity Check
    continuous_edges = 0
    total_transitions = len(pred_route) - 1
    if arcs is not None and total_transitions > 0:
        for i in range(total_transitions):
            v_curr = arcs[pred_route[i]][1]
            u_next = arcs[pred_route[i + 1]][0]
            if v_curr == u_next:
                continuous_edges += 1
        continuity_pct = (continuous_edges / total_transitions) * 100.0
    else:
        continuity_pct = 100.0 if total_transitions == 0 else 0.0

    return {
        "pred_len": n,
        "gt_len": m,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lcs": lcs,
        "lcs_accuracy": lcs_accuracy,
        "continuity_pct": continuity_pct
    }
