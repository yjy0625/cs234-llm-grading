from pydantic import BaseModel
from google import genai

class RubricItem(BaseModel):
    criterion: str
    deduction_applied: bool
    deduction: int
    feedback: str

class GradingResult(BaseModel):
    max_score: int
    overall_summary: str
    results: list[RubricItem] 
    total_score: int          

client = genai.Client()

# The system instruction defines the "logic" of the grader
system_prompt = """
You are a rigorous Academic Grader. Follow these steps exactly:

1. EVALUATE: Compare the Student Response against the Correct Answer and Rubric.
2. LIST EVIDENCE: For each distinct deduction in the Rubric, create a RubricItem. 
   - 'criterion': The specific rule from the rubric.
   - 'deduction_applied': True only if the deduction should be applied
   - 'deduction': The numerical penalty assigned by the rubric if 'deduction_applied' is True. Else the deduction is 0.
   - 'feedback': Explain the reasoning for the deduction or lack of deduction
3. CALCULATE: Get the max_score from the prompt. Set 'total_score' as (max_score minus the sum of all deductions).  
   - Minimum score is 0. 
   - Double-check your math before finalizing the JSON.
4. SUMMARIZE: Write a 1-2 sentence overall feedback for the grading of the question.

Output valid JSON matching the provided schema. Please follow the rubric exactly.
"""

# The specific data from your example
user_content = """
QUESTION:
Explain why the optimal policy for the AI car is not to merge onto the highway. [2 pts]

CORRECT ANSWER KEY:
Merging slows down the human cars. Staying parked allows human cars to maintain max speed, 
resulting in a higher mean velocity for the group than if everyone had to slow down for a merge. 

MAX SCORE:
2

RUBRIC:
- 1 pt deduction: Doesn't explicitly or implicitly mention that merging would slow down the average speed.
- 2 pts deduction: Incorrect or not attempted.

STUDENT RESPONSE:
The optimal policy for the AI car is to not merge, because when it merges it reduces
the mean velocity of all the human-driven cars. Assuming there are enough human cars, the
reduction in their velocity from the merging will be more than the addition to total velocity if
the AI car does merge.
"""

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=user_content,
    config={
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": GradingResult,
    }
)

# Parse and print the structured result
grading = response.parsed
readable_json = grading.model_dump_json(indent=4)
print(readable_json)

# for item in grading.results:
#     print(f"{item.criterion} (Deduction: {item.deduction})")
#     print(f"   Feedback: {item.feedback}")

