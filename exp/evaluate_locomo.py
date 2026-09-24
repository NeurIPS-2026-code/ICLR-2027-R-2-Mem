import json
import os
import re
import math
from collections import Counter, defaultdict

# Calculate metrics

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks, ptoks = tokens(gold), tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0

    gcount, pcount = Counter(gtoks), Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)

    if overlap == 0:
        return 0.0

    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks, ptoks = tokens(gold), tokens(pred)
    if not ptoks:
        return 0.0

    gcount, pcount = Counter(gtoks), Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks)

    if len(ptoks) >= len(gtoks):
        bp = 1.0
    else:
        bp = math.exp(1 - len(gtoks) / len(ptoks)) if ptoks and gtoks else 0.0

    return bp * precision

def compute_metrics_by_category(items, pred_key="summary_answer"):
    agg = defaultdict(list)
    rows = []

    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        pred = ex.get(pred_key, "")

        if isinstance(pred, dict):
            pred = pred.get("answer", "")

        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)

        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "question": ex.get("question", ""),
            "gold": gold,
            "pred": pred,
            "F1": f1,
            "BLEU1": b1
        })

    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        f1_avg = sum(s[0] for s in scores) / len(scores)
        b1_avg = sum(s[1] for s in scores) / len(scores)
        summary.append({
            "category": cat,
            "count": len(scores),
            "F1_avg": f1_avg,
            "BLEU1_avg": b1_avg
        })

    return summary, rows

# ========== muti-conv ==========

def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_multi_conv_evaluation(file_list, title=""):
    all_results = []
    total_iterations = 0

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue

        # print(f"Loading results from: {file_path}")
        data = load_json_file(file_path)

        all_results.extend(data)

        for item in data:
            total_iterations += item.get("iterations", 0)

    if not all_results:
        print(f"[Error] No valid data found for {title}")
        return

    summary, details = compute_metrics_by_category(all_results)

    all_f1 = [r["F1"] for r in details]
    all_b1 = [r["BLEU1"] for r in details]
    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    overall_b1 = sum(all_b1) / len(all_b1) if all_b1 else 0.0

    print("\n" + "=" * 40)
    print("LOCOMO EVALUATION REPORT")
    print("=" * 40)
    for r in summary:
        print(
            f"Category {r['category']}: "
            f"n={r['count']}, F1={r['F1_avg']:.4f}, BLEU1={r['BLEU1_avg']:.4f}"
        )
    print("-" * 40)
    print(
        f"OVERALL: Questions={len(details)}, "
        f"F1={overall_f1:.4f}, BLEU1={overall_b1:.4f}"
    )
    print("=" * 40)

    print(f"Total Iterations: {total_iterations}")
    print(f"Avg Iterations per Question: {total_iterations / len(details):.4f}")

# ========== main ==========

if __name__ == "__main__":
    # evaluate the conv of the locomo
    convs = [30, 41, 42, 43, 44, 47, 48, 49, 50]
    model_size = "7B"

    exp_results_path = "..."
    noexp_results_path = "..."

    exp_file_list = [
        f"{exp_results_path}/{conv}/qa_results.json"
        for conv in convs
    ]

    noexp_file_list = [
        f"{noexp_results_path}/conv-{conv}/qa_results.json"
        for conv in convs
    ]

    print("exp")
    run_multi_conv_evaluation(exp_file_list)

    print("\nno exp")
    run_multi_conv_evaluation(noexp_file_list)