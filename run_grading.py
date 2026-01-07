import argparse
import os
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, Field
from google import genai

MODEL_NAME = "gemini-2.0-flash"


class PartGrading(BaseModel):
    rubric_item: str
    points_awarded: float
    comments: str | None = None


class RawPartResponse(BaseModel):
    rubric_item: str
    comments: str | None = None
    points_deducted: float = Field(..., ge=0.0)


SYSTEM_PROMPT = """
You are a rigorous Academic Grader. Follow these steps exactly:

1. EVALUATE: Compare the Student Response against the Correct Answer and Rubric.
2. LIST EVIDENCE: Reason about the rubric that best applies. You must select exactly one rubric option and copy its text verbatim into 'rubric_item'.
3. DEDUCTION: Output 'points_deducted' from the chosen 'rubric_item' as a nonnegative number (ignore the minus sign) representing how many points to subtract from MAX SCORE.
   - If the response is perfect, points_deducted = 0.
4. JUSTIFY: Provide a concise justification (1–2 sentences) in 'comments' (or null if nothing to add).
5. OUTPUT: Return valid JSON matching the RawPartResponse schema. Double-check arithmetic before replying.
"""


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in the environment before running the grader.")
    return genai.Client(api_key=api_key)


def load_assignment_info(info_path: Path) -> dict:
    with info_path.open("r", encoding="utf-8") as info_file:
        return yaml.safe_load(info_file)


def extract_responses(tex_text: str) -> dict[str, str]:
    pattern = re.compile(r"%%%%% Start of ([^%]+) %%%%%(.*?)%%%%% End of \1 %%%%%", re.DOTALL)
    responses: dict[str, str] = {}
    for label, body in pattern.findall(tex_text):
        responses[label.strip()] = body.strip()
    return responses


def format_rubric(rubric_items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in rubric_items)


def build_user_prompt(question_description: str, part_solution: str, part_points: float, rubric_items: list[str], student_answer: str, part_label: str) -> str:
    rubric_text = format_rubric(rubric_items)
    answer_text = student_answer.strip() or "No response provided."
    return f"""
QUESTION DESCRIPTION:
{question_description}

PART LABEL: {part_label}
MAX SCORE: {part_points}

CORRECT ANSWER SUMMARY:
{part_solution}

RUBRIC OPTIONS (select exactly one and copy it verbatim):
{rubric_text}

STUDENT RESPONSE:
{answer_text}
"""


def clamp_score(value: float, max_points: float) -> float:
    return max(0.0, min(float(max_points), float(value)))


def parse_deduction_from_rubric(rubric_text: str) -> float:
    prefix = rubric_text.split(":", 1)[0]
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", prefix)
    if not m:
        return 0.0
    num = float(m.group())
    return abs(num)


def grade_part(client: genai.Client, prompt: str, max_points: float) -> PartGrading:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": RawPartResponse,
            "temperature": 0.0,
        },
    )
    parsed = response.parsed
    # deduction = parse_deduction_from_rubric(parsed.rubric_item)
    # awarded = clamp_score(max_points - deduction, max_points)
    print(parsed.rubric_item, parsed.points_deducted)
    awarded = clamp_score(max_points - float(parsed.points_deducted), max_points)

    return PartGrading(rubric_item=parsed.rubric_item, points_awarded=awarded, comments=parsed.comments)


def part_label(problem_key: str, part_key: str) -> str:
    numeric = problem_key.lstrip("qQ") or problem_key
    return f"{numeric}({part_key})"


def preview_prompt(tex_name: str, label: str, prompt: str) -> None:
    print(f"===== PROMPT PREVIEW: {tex_name} :: {label} =====")
    print(prompt.strip())
    print("===== END PROMPT =====\n")


def grade_student(tex_path: Path, assignment_info: dict, client: genai.Client | None, output_dir: Path, dry_run: bool) -> None:
    tex_text = tex_path.read_text(encoding="utf-8")
    responses = extract_responses(tex_text)
    parts_output: dict[str, dict] = {}
    total_points = 0.0

    for problem_key, problem_data in assignment_info["problems"].items():
        question_description = problem_data["description"]
        for part_key, part_data in problem_data["parts"].items():
            label = part_label(problem_key, part_key)
            student_answer = responses.get(label, "")
            prompt = build_user_prompt(
                question_description=question_description,
                part_solution=part_data["solution"],
                part_points=part_data["points"],
                rubric_items=part_data["rubric"],
                student_answer=student_answer,
                part_label=label,
            )
            if dry_run:
                preview_prompt(tex_path.name, label, prompt)
                continue
            grading = grade_part(client, prompt, part_data["points"])
            total_points += grading.points_awarded

            part_identifier = f"{problem_key}_{part_key}"
            part_entry = {
                "max_points": float(part_data["points"]),
                "points_awarded": float(grading.points_awarded),
                "rubric_item": grading.rubric_item,
            }
            if grading.comments:
                part_entry["comments"] = grading.comments
            parts_output[part_identifier] = part_entry

    if dry_run:
        return

    student_report = {
        "student_file": tex_path.name,
        "total_points": float(round(total_points, 4)),
        "parts": parts_output,
    }

    output_path = output_dir / f"{tex_path.stem}.yaml"
    output_path.write_text(yaml.safe_dump(student_report, sort_keys=False), encoding="utf-8")
    print(f"Graded {tex_path.name}: {student_report['total_points']} pts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade solutions with Gemini or preview prompts via dry run.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API or writing grades.")
    args = parser.parse_args()

    root = Path(__file__).parent
    assignment_info = load_assignment_info(root / "info.yaml")
    submissions_dir = root / "submissions"
    output_dir = root / "grades"
    output_dir.mkdir(exist_ok=True)

    client = None if args.dry_run else get_client()

    for tex_path in sorted(submissions_dir.glob("*.tex")):
        grade_student(tex_path, assignment_info, client, output_dir, args.dry_run)


if __name__ == "__main__":
    main()

