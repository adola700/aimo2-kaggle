import argparse
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from transformers import set_seed
set_seed(99)
import gc
import time
import warnings
import pandas as pd
import re
import torch
pd.set_option('display.max_colwidth', None)
from datasets import load_dataset
from vllm import LLM, SamplingParams
warnings.simplefilter('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from huggingface_hub import HfFolder
import time
start_time = time.time()
from evaluate import evaluate_all
from collections import Counter
# Save your token locally
HfFolder.save_token("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")

################ Parse Arguments ################
parser = argparse.ArgumentParser(description="Script for fine-tuning and inference with LLM.")
parser.add_argument('--doc_name', type=str, default="final_test_aimo2.csv", help="Path to the input CSV file.")
# parser.add_argument('--llm_model_pth', type=str, default='akh99/DeepSeek-R1-Distill-Qwen-7B-AWQ-tok-norm', help="Path or name of the LLM model.")
parser.add_argument('--llm_model_pth', type=str, default='casperhansen/deepseek-r1-distill-qwen-1.5b-awq', help="Path or name of the LLM model.")
parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use.")
parser.add_argument('--max_model_len', type=int, default=22000, help="Maximum model length for token generation.")
parser.add_argument('--batch_size', type=int, default=10, help="Batch size for inference.")
parser.add_argument('--reps', type=int, default=16, help="Batch size for inference.")
parser.add_argument('--min_p', type=float, default=0.0, help="Minimum cumulative probability for nucleus sampling.")
parser.add_argument('--top_p', type=float, default=0.95, help="Minimum cumulative probability for nucleus sampling.")
parser.add_argument('--temperature', type=float, default=0.6, help="Minimum cumulative probability for nucleus sampling.")
parser.add_argument("--question_column", type=str, default = "problem")
parser.add_argument("--answer_column", type=str, default = "answer")
args = parser.parse_args()

################ Load Dataset and Model ################

doc_name = args.doc_name
df = pd.read_csv(doc_name)

llm_model_pth = args.llm_model_pth
num_gpus = args.num_gpus
max_model_len = args.max_model_len
BS = args.batch_size
min_p = args.min_p
top_p = args.top_p
temperature = args.temperature
question_column = args.question_column
answer_column = args.answer_column
reps = args.reps

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm = LLM(
    llm_model_pth,
    max_model_len=max_model_len,         
    trust_remote_code=True,     
    tensor_parallel_size=num_gpus,      
    gpu_memory_utilization=0.97, 
    # speculative_model = "akh99/DeepSeek-R1-Distill-Qwen-1.5B-tok-norm",
    # num_speculative_tokens = 2,
)

tokenizer = llm.get_tokenizer()

################ Solve function ################
def solve(problems, temperature=temperature, rep=reps):
    sampling_params = SamplingParams(
        temperature=temperature,              # Controls randomness in generation: higher values (e.g., 1.0) produce more diverse output.
        min_p=min_p,        
        top_p=top_p, # Minimum cumulative probability for nucleus sampling, filtering out unlikely tokens.
        skip_special_tokens=False,     
        max_tokens=max_model_len,             # Sets a very high limit for token generation to handle longer outputs.
        stop_token_ids = [151649]
    )
    problems_list = []
    for problem in problems:
        for i in range(rep):
            problems_list.append(problem)

    list_of_messages = [
        [
            {"role": "system", "content": "Please think step-by-step and give final answer using \\boxed{}, after taking modulo 1000."},
            {"role": "user", "content": problem}
        ] for problem in problems_list
    ]    

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )

    return [request_output[i].outputs[0].text for i in range(len(problems_list)) ]

def extract_boxed_texts(text):
    pattern = r'\\boxed{((?:[^{}]|{[^{}]*})+)}'
    matches = re.findall(pattern, text)
    if not matches:
        return -1000
    ans = []
    for content in matches:
        if content.isdigit():
            num = int(content)
            ans.append(num)

    if len(ans) > 0: return ans[-1]
    return matches[-1]        
    
################ Final inference ################
df = df[60:70]
from tqdm import tqdm
start = 0
end = len(df)

outputs = []
answers = []
lengths = []

for i in tqdm(range(start, end, BS)):
    try:
        solns = solve(list(df[question_column][i:i+BS]))
        outputs.extend(solns)
        answers.extend([extract_boxed_texts(soln) for soln in solns])
        lengths.extend([len(tokenizer(soln)["input_ids"]) for soln in solns])
        print(answers)
    except:
        print("exception occured at: ", i)

################ Store results ################
duration = time.time() - start_time

outputs_batched = []
answers_batched = []
lengths_batched = []

N = len(outputs)
for i in range(0, N, reps):
    outputs_batch = []
    answers_batch = []
    lengths_batch = []
    for j in range(reps):
        outputs_batch.append(outputs[i+j])
        answers_batch.append(answers[i+j])
        lengths_batch.append(lengths[i+j])
        
    outputs_batched.append(outputs_batch)
    answers_batched.append(answers_batch)
    lengths_batched.append(lengths_batch)

df["outputs"] = outputs_batched
df["gen_answers"] = answers_batched
df["output_lengths"] = lengths_batched
df.to_csv(f"{doc_name}_full_preds.csv", index=None)

targets = list(df[answer_column])

eval_obj = evaluate_all(answers_batched, lengths_batched, targets)
eval_obj.print_all_metrics()

print(f"duration = {duration}")
print(f"Average tokens consumed = {sum(lengths)/len(lengths)}")
