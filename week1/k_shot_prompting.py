import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You are a string processing engine.

Task: Given an input of a string, you should reverse the string character-by-character, and then output the result.

Ensure the string reversal is at the character-level, not the word-level. You should treat the input as a list of characters.
Some strings will be mashed together with no spaces. For example, mysteryman (instead of mystery man). In that case, you should reverse as namyretsym and NOT manmystery (TREAT AS ONE FULL LIST OF CHARACTERS).

Here are some examples of the task you need to do.

EXAMPLES:

INPUT: httpport
OUTPUT: tropptth

INPUT: status
OUTPUT: sutats

INPUT: httpstatus
OUTPUT: sutatsptth


INPUT: ballatro
OUTPUT: ortallab

INPUT: wagwan
OUTPUT: nawgaw

INPUT: baklava
OUTPUT: avalkab

INPUT: mysteryman
OUTPUT: namyretsym

INPUT: tonight
OUTPUT: thginot
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)
