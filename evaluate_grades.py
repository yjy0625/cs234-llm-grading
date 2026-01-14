#!/usr/bin/env python3
"""Evaluate grader outputs against synthetic ground-truth metadata.

Example:
  python evaluate_grades.py --metadata generated_submissions/metadata.yaml --grades-dir grades
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate grades against metadata")
    p.add_argument("--metadata", type=Path, default=Path("generated_submissions/metadata.yaml"))
    p.add_argument("--grades-dir", type=Path, default=Path("grades"))
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def almost_equal(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= EPS


def main():
    args = parse_args()
    meta = load_yaml(args.metadata)
    submissions = meta.get("submissions", {})

    total_parts = 0
    total_correct = 0
    per_part_stats: Dict[str, Dict[str, float]] = {}
    total_score_error = 0.0
    count_scores = 0
    total_part_abs_error = 0.0
    missing_grade_files = 0

    for meta_filename, meta_entry in submissions.items():
        stem = Path(meta_filename).stem
        grade_path = args.grades_dir / f"{stem}.yaml"
        if not grade_path.exists():
            missing_grade_files += 1
            continue
        grader = load_yaml(grade_path)
        # total score
        gt_total = float(meta_entry.get("total_points", 0.0))
        grader_total = float(grader.get("total_points", 0.0))
        total_score_error += abs(gt_total - grader_total)
        count_scores += 1

        # iterate parts from metadata
        questions = meta_entry.get("questions", {})
        for qk, qinfo in questions.items():
            parts = qinfo.get("parts", {})
            for part_key, part_info in parts.items():
                gt_points = float(part_info.get("points_awarded", 0.0))
                grader_part_key = f"{qk}_{part_key}"
                grader_parts = grader.get("parts", {})
                if grader_part_key not in grader_parts:
                    # treat missing part as zero awarded and count as error
                    gr_points = 0.0
                else:
                    gr_points = float(grader_parts[grader_part_key].get("points_awarded", 0.0))
                total_parts += 1
                if almost_equal(gt_points, gr_points):
                    total_correct += 1
                # per-part stats key
                per_key = grader_part_key
                if per_key not in per_part_stats:
                    per_part_stats[per_key] = {"total": 0.0, "correct": 0.0, "abs_error_sum": 0.0}
                per_part_stats[per_key]["total"] += 1.0
                if almost_equal(gt_points, gr_points):
                    per_part_stats[per_key]["correct"] += 1.0
                per_part_stats[per_key]["abs_error_sum"] += abs(gt_points - gr_points)
                total_part_abs_error += abs(gt_points - gr_points)

    # report
    print("Evaluation results:")
    print(f"Submissions in metadata: {len(submissions)}")
    print(f"Missing grade files: {missing_grade_files}")
    if total_parts == 0:
        print("No parts evaluated.")
        return
    overall_accuracy = total_correct / total_parts
    avg_l1_total = (total_score_error / count_scores) if count_scores else float("nan")
    avg_l1_per_part = total_part_abs_error / total_parts

    print()
    print(f"Overall parts accuracy: {overall_accuracy:.4f} ({int(total_correct)}/{total_parts})")
    print(f"Average L1 error in total score: {avg_l1_total:.4f} (over {count_scores} graded submissions)")
    print(f"Average L1 error per part: {avg_l1_per_part:.4f}")

    print()
    print("Per-part stats:")
    # sort parts for stable output
    for part_key in sorted(per_part_stats.keys()):
        stats = per_part_stats[part_key]
        total = int(stats["total"])
        correct = int(stats["correct"])
        abs_err = stats["abs_error_sum"]
        acc = correct / total if total else 0.0
        avg_err = abs_err / total if total else 0.0
        print(f"- {part_key}: accuracy={acc:.4f} ({correct}/{total}), avg_l1={avg_err:.4f}")


if __name__ == "__main__":
    main()
