VERIFY_CORRECTNESS_PROMPT = """Given two answers to a question, check if they are the same. Without any explanation, respond with 1 if they are the same, otherwise 0. If one answer has multiple sub solutions and the other answer list only one of them , consider it an Yes.
Examples:
Answer1: \\dfrac{12}{5}, Answer2: 12/5 --> 1
Answer1: 7/3, Answer2: 3/7 --> 0
Answer1: 1/2, Answer2: 0.5 --> 1
Answer1: 4, Answer2: 4.0 --> 1
Answer1: 4,5, Answer2: 4.0 --> 1 (even if one subsolution match consider it correct - 1)
Input: Answer1: <answer1>, Answer2: <answer2> -> 0 or 1.

Input: """


GET_ANSWER_PROMPT = "Given a long/partial solution, Just return the final answer without any explanation (If no final answer present return -1000 ).\n\n Solution:\n\n" 

CONJUCTION_WORDS = ["Wait", "Alternatively", "But", "Now", "Therefore", "So", "Hmm"]

HINTS = {
    "complex calculations hint": "Wait, I can use Python to perform complex calculations for this problem.```python\n",
    "self reflection hint": "Wait, I can use Python to check if my approach is correct and refine it, if necessary.```python\n",
    "check logic hint": "Maybe Python can assist in ensuring our logical deductions are sound.```python\n",
    "alternative method hint": "Alternatively, I can use Python to explore an alternative method for solving this problem.```python\n",
    "general hint": "Wait, using python here may be a good idea.```python\n",
    "explore deeply hint": "Wait, I can explore deeply about this problem through python tools.```python\n"
}

PLACE_HINT = """Task: Given a question and a step-by-step Chain-of-Thought (CoT) response, your goal is to strategically insert one of the predefined code hints to accelerate problem-solving or solve the question.

Guidelines for Hint Placement:

Initial Thinking Phase:
Do not insert hints in the first 2000–3000 tokens (let the model reason independently first).

Hint Selection Criteria:
Use "complex calculations hint" for math-heavy steps (e.g., solving equations, expanding expressions).
Use "self reflection hint" when it is optimal to double-checks its reasoning.
Use "check logic hint" for verifying deductions or edge cases.
Use "alternative method hint" if it is optimal to use a different approach.
Use "explore deeply hint" for open-ended investigation (e.g., testing hypotheses).
Default to "general hint" if no specific category fits.

hint_type in output should be from one of these types.

Output Format(only json without any explanation - follow this format):
```json
{
    "line_number": int,  # Step number where hint is most impactful.
    "hint_type": str     # Key from HINTS dict (e.g., "complex calculations hint").
}
```

Example:
Input:
For a response where the model struggles with solving a quadratic equation at step 15:
Output:
{"line_number": 15, "hint_type": "complex calculations hint"}

Now I told you all instructions. Please consider below input and give output in Json format.

Input:
"""


# Existing code already:
# If there is code already existing in between reasoning steps, make sure you choose any step before it - only if it makes full sense, else please choose a step after it following the guidelines.(which increases chances of success)