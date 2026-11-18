#!/usr/bin/env python3
"""
Main Map Matching Pipeline & Benchmark Runner
=============================================
Usage:
    python3 run_pipeline.py --all
    python3 run_pipeline.py --case 00000000
    python3 run_pipeline.py --dir "Student folder/student_test"
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.map_matcher import MapMatcher
from src.evaluator import evaluate_route


def main():
    parser = argparse.ArgumentParser(description="Map Matching Pipeline Runner")
    parser.add_argument("--dir", default="Student folder/student_test", help="Test directory containing cases")
    parser.add_argument("--case", default=None, help="Specific case ID to process (e.g. 00000000)")
    parser.add_argument("--all", action="store_true", default=True, help="Run all cases in directory")
    parser.add_argument("--out", default="predictions", help="Directory to save predicted .route files")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, args.dir)
    out_dir = os.path.join(project_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        sys.exit(1)

    matcher = MapMatcher(sigma_z=7.0, beta=10.0, search_radius=50.0, top_k=5)

    cases = []
    if args.case:
        cases = [args.case]
    else:
        cases = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    print("=" * 80)
    print(f"       MAP MATCHING BENCHMARK PIPELINE ({len(cases)} Cases)")
    print("=" * 80)
    print(f"{'Case ID':10s} | {'Points':7s} | {'Pred':5s} | {'GT':5s} | {'Prec':6s} | {'Recall':6s} | {'F1':6s} | {'LCS Acc':7s} | {'Cont':5s} | {'Time':6s}")
    print("-" * 80)

    results = []
    total_time = 0.0

    for case_id in cases:
        case_path = os.path.join(test_dir, case_id)
        try:
            nodes, arcs, track, gt_route = matcher.load_case(case_path, case_id)
            pred_route, stats = matcher.match(nodes, arcs, track)

            # Save prediction
            pred_file = os.path.join(out_dir, f"{case_id}.route")
            with open(pred_file, "w") as pf:
                for arc_id in pred_route:
                    pf.write(f"{arc_id}\n")

            total_time += stats["elapsed_seconds"]

            if gt_route is not None:
                metrics = evaluate_route(pred_route, gt_route, arcs)
                results.append((case_id, stats, metrics))
                print(f"{case_id:10s} | {stats['raw_track_points']:7d} | {metrics['pred_len']:5d} | {metrics['gt_len']:5d} | "
                      f"{metrics['precision']*100:5.1f}% | {metrics['recall']*100:5.1f}% | {metrics['f1']*100:5.1f}% | "
                      f"{metrics['lcs_accuracy']*100:6.1f}% | {metrics['continuity_pct']:4.0f}% | {stats['elapsed_seconds']:5.2f}s")
            else:
                print(f"{case_id:10s} | {stats['raw_track_points']:7d} | {len(pred_route):5d} | {'N/A':5s} | {'-':6s} | {'-':6s} | {'-':6s} | {'-':7s} | {'-':5s} | {stats['elapsed_seconds']:5.2f}s")

        except Exception as e:
            print(f"{case_id:10s} | ERROR: {e}")

    print("-" * 80)
    if results:
        avg_prec = sum(r[2]['precision'] for r in results) / len(results) * 100.0
        avg_rec = sum(r[2]['recall'] for r in results) / len(results) * 100.0
        avg_f1 = sum(r[2]['f1'] for r in results) / len(results) * 100.0
        avg_lcs = sum(r[2]['lcs_accuracy'] for r in results) / len(results) * 100.0
        avg_cont = sum(r[2]['continuity_pct'] for r in results) / len(results)

        print(f"{'AVERAGE':10s} | {'':7s} | {'':5s} | {'':5s} | {avg_prec:5.1f}% | {avg_rec:5.1f}% | {avg_f1:5.1f}% | {avg_lcs:6.1f}% | {avg_cont:4.0f}% | {total_time:5.2f}s")
        print("=" * 80)

        # Generate Markdown Report
        report_path = os.path.join(project_root, "reports/benchmark_report.md")
        with open(report_path, "w") as rf:
            rf.write("# Map Matching Benchmark Report\n\n")
            rf.write(f"- **Evaluated Cases**: {len(results)}\n")
            rf.write(f"- **Total Time**: {total_time:.2f} seconds\n")
            rf.write(f"- **Average Recall**: {avg_rec:.2f}%\n")
            rf.write(f"- **Average Precision**: {avg_prec:.2f}%\n")
            rf.write(f"- **Average F1 Score**: {avg_f1:.2f}%\n")
            rf.write(f"- **Average LCS Accuracy**: {avg_lcs:.2f}%\n")
            rf.write(f"- **Average Topological Continuity**: {avg_cont:.1f}%\n\n")
            rf.write("## Case-by-Case Breakdown\n\n")
            rf.write("| Case ID | Track Points | Predicted Arcs | Ground Truth Arcs | Precision | Recall | F1 Score | LCS Accuracy | Continuity | Latency |\n")
            rf.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for case_id, stats, m in results:
                rf.write(f"| `{case_id}` | {stats['raw_track_points']} | {m['pred_len']} | {m['gt_len']} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['lcs_accuracy']*100:.2f}% | {m['continuity_pct']:.1f}% | {stats['elapsed_seconds']:.2f}s |\n")

        print(f"\nSaved detailed benchmark report to: {report_path}")
        print(f"Saved predicted .route files to: {out_dir}/")


if __name__ == "__main__":
    main()
