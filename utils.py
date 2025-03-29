import re
import openai
import os
from prompts import *
from openai import OpenAI
import concurrent.futures
import pandas as pd
from globals import *
import json
from transformers import AutoTokenizer
import random
# Tokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
# ---------------------------------------------------------------------
def parallelize(fn, *args):
    """
    Parallelizes the execution of a function over multiple lists of arguments.
    
    Parameters:
        fn (callable): The function to be executed.
        *args (list): One or more lists of arguments. The i-th elements from each list
                      are passed together to fn.
                      
    Returns:
        list: A list of results from applying fn on each set of arguments.
    """
    if not args:
        return []
    
    # Ensure all lists have the same length
    n = len(args[0])
    if any(len(lst) != n for lst in args):
        raise ValueError("All input lists must have the same length.")

    results = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        # Submit each job with the corresponding elements from all lists.
        future_to_index = {
            executor.submit(fn, *[arg[i] for arg in args]): i
            for i in range(n)
        }
        # Retrieve results as they complete.
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = str(e)
                print(str(e))
    return results
# ---------------------------------------------------------------------
# REGEX FUNCTIONS

def return_wrongs_df_only_aime_format(df, output_column = "outputs", answer_column = "answer"):
    df_deep_copy = df.copy(deep=True)
    answers = [-1000 if extract_boxed_texts(item) is None else extract_boxed_texts(item) for item in list(df[output_column])  ]
    df_deep_copy["gen_answers"] = answers
    df_deep_copy["answer"] = df_deep_copy["answer"].apply(lambda x: str(x))
    return df_deep_copy[~(df_deep_copy["answer"] == df_deep_copy["gen_answers"])].reset_index(drop = True)

def get_length(text):
    return len(tok(text, add_special_tokens = False)["input_ids"])

def get_truncate(text, max_length = 8192):
        all_tokens = tok(text, add_special_tokens = False)["input_ids"]
        all_tokens = all_tokens[:max_length]
        res = tok.decode(all_tokens)
        res = "\n\n".join(res.split("\n\n")[:-1])
        return res
    
def extract_and_number_lines(text, get_count=False):
    lines = text.split("\n\n")
    result = []
    count = 1
    for line in lines:
        if line.startswith(("Wait", "Alternatively", "But","Now")):
            result.append(f"{count}. {line}")
            count += 1
        else:
            result.append(f"{line}")
    if get_count:
        return "\n\n".join(result), count - 1 
    else:
        return "\n\n".join(result) 

def get_truncated_from_line_number(r, line_number, hint_type):
    lines = r.split("\n\n")
    result = []
    count = 1
    for line in lines:
        if line.startswith(("Wait", "Alternatively", "But","Now")):
            if count == line_number:
                return "\n\n".join(result) + "\n\n" + HINTS[hint_type]
            result.append(f"{line}")  
            count += 1
        else:
            result.append(f"{line}")
    return "\n\n".join(result) + "\n\n" + HINTS[hint_type]

def get_full_prompt_to_place_hints(q, r):
    truncated_r = get_truncate(r)
    full_input = PLACE_HINT +"QUESTION: \n" + q + "\n\nMODEL PARTIAL RESPONSE:\n" + extract_and_number_lines(truncated_r) +  "\n\n\nOUTPUT(in json format): " 
    return full_input

def extract_between_braces(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return eval(input_string[start_index:end_index + 1])["line_number"], eval(input_string[start_index:end_index + 1])["hint_type"]
    return None

def get_random_sample(n):
    import numpy as np

    mean = (n + 1) / 2
    sigma = (n - 1) / 6  # About 3 sigma covers almost the whole range [1, n]
    
    # Keep sampling until a valid point in [1, n] is obtained.
    while True:
        point = int(np.round(np.random.normal(mean, sigma)))
        if 1 <= point <= n:
            return point

def insert_hint(q, r, model_p = 0.0):
    full_prompt = get_full_prompt_to_place_hints(q, r)
    strategy = random.choices(["model", "random"], weights=[model_p, 1 - model_p], k = 1)[0]
    if strategy == "model":
        client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=random.choices([DEEP_INFRA_API_KEY,DEEP_INFRA_API_KEY_2, DEEP_INFRA_API_KEY_3], weights=[1/3,1/3, 1/3], k = 1)[0]
        )
        
        max_tokens = 50
        chat_completion_res = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages= [{"role": "user","content": full_prompt}],
            stream=False,
            max_tokens=max_tokens,
            temperature=0.6,
        )
        # print(chat_completion_res.choices[0].message.content)
        out = extract_between_braces(chat_completion_res.choices[0].message.content)
        if out is None:
            return out
        line_number, hint_type = out
    else:
        _, lines_count = extract_and_number_lines(r, True)
        if lines_count == 0:
            return None
        if lines_count == 1:
            return None
        elif lines_count == 2:
            line_number = 2
        else:    
            line_number,  = get_random_sample(lines_count), 
        hint_idx = random.choices([0,1,2,3,4,5])[0]
        hint_type = list(HINTS.keys())[hint_idx]

    return get_truncated_from_line_number(r, line_number, hint_type)
# ------------------------------------------------------------------------
def extract_boxed_texts(text):
    # Search for either "\boxed{" or "oxed{" in the text.
    # Adjust the pattern as needed if you have other variants to capture.
    pattern = r'(\\boxed\{|oxed\{)'
    match = re.search(pattern, text)
    if not match:
        return None

    # Start parsing right after the matched sequence.
    i = match.end()
    depth = 1
    content = []

    while i < len(text) and depth > 0:
        char = text[i]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                i += 1
                break
        content.append(char)
        i += 1

    if depth != 0:
        return None

    return ''.join(content)

def extract_before_token(input_str, token_string):
    # Find the position of "</think>"
    position = input_str.find(token_string)
    if position != -1:
        # Return everything before "</think>"
        return input_str[:position]
    else:
        # If "</think>" is not found, return the original string or handle it
        return input_str

def extract_python_code(input_str):
    # Regular expression to find content between ```python and ```
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, input_str, re.DOTALL)
    return matches[-1]
# ---------------------------------------------------------------------
def verify_correctness(gen_answer, actual_answer):
#     print("running")
    from openai import OpenAI
    client = OpenAI(
        base_url="https://conductor.arcee.ai/v1",
        api_key=random.choices([CONDUCTOR_API_KEY, CONDUCTOR_API_KEY_2, CONDUCTOR_API_KEY_3])[0]
    )

    # client = OpenAI(
    #     base_url="http://localhost:8000/v1",
    #     api_key="None",
    # )
    
    max_tokens = 10
    chat_completion_res = client.chat.completions.create(
        # model="stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
        model="virtuoso-large",
        messages= [{"role": "user","content": VERIFY_CORRECTNESS_PROMPT + "Answer1: " + str(gen_answer) + "\nAnswer2: " + str(actual_answer)}],
        stream=False,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # print(chat_completion_res.choices[0].message.content)
    return chat_completion_res.choices[0].message.content


def get_answer(solution):
    solution = solution[-350:]
    # print(VERIFY_CORRECTNESS_PROMPT + str(gen_answer) + ", "+ str(actual_answer))
    from openai import OpenAI
    client = OpenAI(
        base_url="https://conductor.arcee.ai/v1",
        api_key=random.choices([CONDUCTOR_API_KEY, CONDUCTOR_API_KEY_2, CONDUCTOR_API_KEY_3])[0]
    )
    # client = OpenAI(
    #     base_url="http://localhost:8000/v1",
    #     api_key="None",
    # )
    
    max_tokens = 50
    chat_completion_res = client.chat.completions.create(
        model="virtuoso-large",
        # model="stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
        messages= [{"role": "user","content": GET_ANSWER_PROMPT + solution + "Final answer: \n"}],
        stream=False,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # print(chat_completion_res.choices[0].message.content)
    return chat_completion_res.choices[0].message.content
# ------------------------------- GET COMPLETIONS --------------------------------------
def get_extra_completion(problem, partial_response, max_tokens = 8192, stop_tokens_list=["```", "</think>"]):
#     print("run started")
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="None"
    )
    
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    max_tokens = max_tokens
    
    chat_completion_res = client.completions.create(
        model=model,
        prompt='<｜begin▁of▁sentence｜>Please think step-by-step and give final answer using \\boxed{}<|User|>'+problem+'<|Assistant|><think>' + partial_response,
        stream=False,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p = 0.95,
        stop = stop_tokens_list
    )
    return chat_completion_res.choices[0].text


def get_code_completion(problem, partial_response, max_tokens = 2048, stop_tokens_list=["```", "</think>"]):
    # print("run started")
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="None",
    )
    
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    max_tokens = max_tokens
    
    chat_completion_res = client.completions.create(
        model=model,
        prompt='<｜begin▁of▁sentence｜>Please think step-by-step and give final answer using \\boxed{}<|User|>'+problem+'<|Assistant|><think>' + partial_response,
        stream=False,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p = 0.95,
        stop = stop_tokens_list
    )
    return chat_completion_res.choices[0].text