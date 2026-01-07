# LLM Grading for CS 234 HW 1

## Setup

- Student answers should be filled out following the template `template_llm.tex`
- Submissions should be placed in the `submissions/` folder as tex files
- Install the Google Gen AI api: `pip install google-genai`
- Set Google API Key: `export GOOGLE_API_KEY=[your Google AI Studio API key]`
- Set up questions and rubrics in `info.yaml`. Make sure to format question text so that it doesn't have special characters, endlines, and single `\` signs (they result in errors when the file is read).

## Running Grading

First, try a dry run to see if the prompts look alright: `python run_grading.py --dry-run`

After you are ready for an actual run, execute `python run_grading.py`. Outputs will be saved to a `grades/` folder.
