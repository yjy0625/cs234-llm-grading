#!/usr/bin/env python3
from __future__ import annotations

"""Generate rubric-stratified synthetic .tex submissions via OpenAI.

Run: python generate_synthetic_submissions.py --output-dir synthetic_runs --num-variants 10
"""

import argparse
import os
import random
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

SYSTEM_PROMPT = (
    "You write realistic student answers that match requested rubric quality. "
    "Never mention rubrics, scoring, or instructions. Only output the required blocks."
)


@dataclass
class PartSpec:
    key: str
    label: str
    max_points: float
    solution: str
    rubric: List[str]


@dataclass
class QuestionSpec:
    key: str
    numeric_label: str
    description: str
    parts: List[PartSpec]

    @property
    def display_name(self) -> str:
        return f"Question {self.numeric_label}"


def parse_args() -> argparse.Namespace:
    default_info = Path(__file__).with_name("info.yaml")
    parser = argparse.ArgumentParser(
        description="Generate synthetic submissions that cover rubric strata via the OpenAI API.",
    )
    parser.add_argument("--info-path", type=Path, default=default_info, help="Path to info.yaml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for .tex files")
    parser.add_argument("-n", "--num-variants", type=int, required=True, help="Number of submissions to generate")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for rubric shuffling")
    parser.add_argument("--prefix", default="synthetic_", help="Filename prefix for generated .tex files")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per API call")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and emit placeholder text")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent requests per question (ignored for --dry-run)",
    )
    return parser.parse_args()


def load_assignment(info_path: Path) -> Dict[str, Any]:
    if not info_path.exists():
        raise FileNotFoundError(f"info file not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "problems" not in data:
        raise ValueError("info.yaml must contain a 'problems' key")
    return data


def build_questions(info: Dict[str, Any]) -> List[QuestionSpec]:
    questions: List[QuestionSpec] = []
    for problem_key, problem_data in info["problems"].items():
        numeric = problem_key.lstrip("qQ") or problem_key
        description = str(problem_data.get("description", "")).strip()
        parts_data = problem_data.get("parts", {})
        part_specs: List[PartSpec] = []
        for part_key, part_data in parts_data.items():
            label = f"{numeric}({part_key})"
            part_specs.append(
                PartSpec(
                    key=part_key,
                    label=label,
                    max_points=float(part_data.get("points", 0)),
                    solution=str(part_data.get("solution", "")).strip(),
                    rubric=[str(item) for item in part_data.get("rubric", [])],
                )
            )
        questions.append(
            QuestionSpec(
                key=problem_key,
                numeric_label=numeric,
                description=description,
                parts=part_specs,
            )
        )
    return questions


def distribute_targets(num_variants: int, rubric_size: int, rng: random.Random) -> List[int]:
    if rubric_size <= 0:
        raise ValueError("Each part must include at least one rubric item")
    base = num_variants // rubric_size
    remainder = num_variants % rubric_size
    if remainder:
        print(
            f"Warning: cannot evenly split {num_variants} samples across {rubric_size} rubric items;",
            "distribution will differ by at most 1.",
            file=sys.stderr,
        )
    counts = [base] * rubric_size
    for idx in range(remainder):
        counts[idx] += 1
    assignments: List[int] = []
    for rubric_idx, count in enumerate(counts):
        assignments.extend([rubric_idx] * count)
    rng.shuffle(assignments)
    return assignments


def build_target_plan(questions: List[QuestionSpec], num_variants: int, rng: random.Random) -> Dict[str, Dict[str, List[int]]]:
    plan: Dict[str, Dict[str, List[int]]] = {}
    for question in questions:
        part_plan: Dict[str, List[int]] = {}
        for part in question.parts:
            part_plan[part.key] = distribute_targets(num_variants, len(part.rubric), rng)
        plan[question.key] = part_plan
    return plan


def build_targets_for_variant(question: QuestionSpec, part_plan: Dict[str, List[int]], idx: int) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for part in question.parts:
        rubric_idx = part_plan[part.key][idx]
        targets.append({
            "part": part,
            "target_index": rubric_idx,
            "target_rubric": part.rubric[rubric_idx],
        })
    return targets


def format_prompt(question: QuestionSpec, targets: List[Dict[str, Any]]) -> str:
    part_sections: List[str] = []
    for target in targets:
        part = target["part"]
        rubric_text = "\n".join(f"- {item}" for item in part.rubric)
        section = textwrap.dedent(
            f"""
            PART: {part.label}
            Max points: {part.max_points}
            Correct answer summary:
            {part.solution}

            Rubric options:
            {rubric_text}

            TARGET RUBRIC FOR THIS PART:
            {target['target_rubric']}
            """
        ).strip()
        part_sections.append(section)
    guidelines = textwrap.dedent(
        f"""
        You must produce one response block per part using this exact structure:
        %%%%% Start of <part label> %%%%%
        <answer text>
        %%%%% End of <part label> %%%%%

        Output the blocks in the order listed below. Leave a single blank line between blocks.
        Do not restate the question or mention rubrics.
        """
    ).strip()
    prompt_parts = [f"{question.display_name} description:", question.description, guidelines]
    prompt_parts.extend(part_sections)
    return "\n\n".join(part for part in prompt_parts if part)


def create_client(dry_run: bool):
    if dry_run:
        return None
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai package to run this script without --dry-run") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment before running this script")
    return OpenAI(api_key=api_key)


def call_model(client, model: str, user_prompt: str, temperature: float, max_retries: int, rng: random.Random) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            backoff = (2 ** (attempt - 1)) + rng.uniform(0, 0.5)
            print(f"API error ({exc}); retrying in {backoff:.1f}s", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def process_variant(
    *,
    question: QuestionSpec,
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
    client,
    seed_data: tuple[Any, ...],
    idx: int,
) -> tuple[int, Dict[str, str]]:
    local_rng = random.Random(seed_data)
    raw_text = call_model(client, model, prompt, temperature, max_retries, local_rng)
    rendered = parse_and_render(question, raw_text)
    return idx, rendered


def format_block(label: str, body: str) -> str:
    content = body.strip() or "No response provided."
    return f"%%%%% Start of {label} %%%%%\n{content}\n%%%%% End of {label} %%%%%"


def extract_responses(tex_text: str) -> Dict[str, str]:
    pattern = re.compile(r"%%%%% Start of ([^%]+) %%%%%(.*?)%%%%% End of \1 %%%%%", re.DOTALL)
    responses: Dict[str, str] = {}
    for match in pattern.finditer(tex_text):
        label = match.group(1).strip()
        body = match.group(2).strip()
        responses[label] = body
    return responses


def parse_and_render(question: QuestionSpec, raw_text: str) -> Dict[str, str]:
    parsed = extract_responses(raw_text)
    rendered: Dict[str, str] = {}
    for part in question.parts:
        if part.label not in parsed:
            snippet = raw_text.strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            raise ValueError(
                f"Missing block for {part.label} in response. Received text:\n{snippet}"
            )
        rendered[part.label] = format_block(part.label, parsed[part.label])
    return rendered


def clamp_score(value: float, max_points: float) -> float:
    return max(0.0, min(max_points, value))


def parse_deduction_from_rubric(rubric_text: str) -> float:
    prefix = rubric_text.split(":", 1)[0]
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", prefix)
    if not match:
        return 0.0
    return abs(float(match.group()))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def main() -> None:
    args = parse_args()
    if args.num_variants <= 0:
        raise SystemExit("--num-variants must be positive")
    rng = random.Random(args.seed)
    info = load_assignment(args.info_path)
    questions = build_questions(info)
    if not questions:
        raise SystemExit("No questions found in info.yaml")
    target_plan = build_target_plan(questions, args.num_variants, rng)
    client = create_client(args.dry_run)
    ensure_output_dir(args.output_dir)
    question_segments: Dict[str, List[Dict[str, str]]] = {}

    for question in questions:
        log(f"Generating responses for {question.key} ({len(question.parts)} parts)", args.quiet)
        part_plan = target_plan[question.key]
        if args.dry_run:
            question_segments[question.key] = []
            for idx in range(args.num_variants):
                targets = build_targets_for_variant(question, part_plan, idx)
                prompt = format_prompt(question, targets)
                print(f"===== DRY-RUN PROMPT: {question.key} variant {idx + 1} =====")
                print(prompt)
                print("===== END DRY-RUN PROMPT =====")
                breakpoint()
                raw_text = "\n\n".join(
                    format_block(t["part"].label, f"Placeholder for {t['target_rubric']}") for t in targets
                )
                rendered = parse_and_render(question, raw_text)
                question_segments[question.key].append(rendered)
                if not args.quiet:
                    log(f"  - {question.key} variant {idx + 1}/{args.num_variants}", args.quiet)
            continue

        question_segments[question.key] = [None] * args.num_variants  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for idx in range(args.num_variants):
                targets = build_targets_for_variant(question, part_plan, idx)
                prompt = format_prompt(question, targets)
                seed_data = (args.seed, question.key, idx)
                futures.append(
                    executor.submit(
                        process_variant,
                        question=question,
                        prompt=prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        client=client,
                        seed_data=seed_data,
                        idx=idx,
                    )
                )
            for future in as_completed(futures):
                idx, rendered = future.result()
                question_segments[question.key][idx] = rendered
                if not args.quiet:
                    log(f"  - {question.key} variant {idx + 1}/{args.num_variants}", args.quiet)

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "num_variants": args.num_variants,
        "submissions": {},
    }

    for idx in range(args.num_variants):
        filename = f"{args.prefix}{idx:05d}.tex"
        output_path = args.output_dir / filename
        submission_blocks: List[str] = []
        submission_meta: Dict[str, Any] = {"questions": {}, "total_points": 0.0}
        for question in questions:
            rendered_parts = question_segments[question.key][idx]
            ordered_parts = [rendered_parts[part.label] for part in question.parts]
            submission_blocks.append("\n\n".join(ordered_parts))
            question_meta: Dict[str, Any] = {"label": question.numeric_label, "parts": {}, "points_awarded": 0.0}
            for part in question.parts:
                rubric_idx = target_plan[question.key][part.key][idx]
                rubric_item = part.rubric[rubric_idx]
                deduction = parse_deduction_from_rubric(rubric_item)
                awarded = clamp_score(part.max_points - deduction, part.max_points)
                question_meta["parts"][part.key] = {
                    "label": part.label,
                    "max_points": part.max_points,
                    "target_rubric": rubric_item,
                    "points_awarded": round(awarded, 4),
                }
                question_meta["points_awarded"] += awarded
            submission_meta["questions"][question.key] = {
                "numeric_label": question.numeric_label,
                "points_awarded": round(question_meta["points_awarded"], 4),
                "parts": question_meta["parts"],
            }
            submission_meta["total_points"] += question_meta["points_awarded"]
        submission_meta["total_points"] = round(submission_meta["total_points"], 4)
        output_path.write_text("\n\n".join(submission_blocks).strip() + "\n", encoding="utf-8")
        metadata["submissions"][filename] = submission_meta
        log(f"Wrote {output_path}", args.quiet)

    metadata_path = args.output_dir / "metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    log(f"Metadata saved to {metadata_path}", args.quiet)


if __name__ == "__main__":
    main()
