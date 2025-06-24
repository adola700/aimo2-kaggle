################ Imports ################
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd

################ Load Model and Tokenizer ################
model_id = "Qwen/QwQ-32B-Preview"  # You can use "EleutherAI/gpt-neo-125M", etc.
train_file = "hard_long_cots_wo_math.csv"
# Define the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype= torch.bfloat16, attn_implementation = "flash_attention_2", device_map = "auto")
LR = 7e-6 

# Ensure the tokenizer adds a padding token (important for batch processing)
if tokenizer.pad_token is None:
    print("fixing pad token")
    tokenizer.pad_token = tokenizer.eos_token
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

################ Load Dataset amd preprocess ################
ds = Dataset.from_pandas(pd.read_csv(train_file))
train_test_split = ds.train_test_split(test_size=0.06)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Preprocessing function
def preprocess_function(examples):
    question = examples["input"]
    answer = examples["label"]
    
    messages = [
    {"role": "system", "content": "Please think step-by-step and give final answer using \boxed{}"},
    {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    question_ids = tokenizer(text, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    
    input_ids = question_ids + answer_ids + [tokenizer.eos_token_id]
    attention_mask = len(input_ids)*[1]
    labels = [-100]*len(question_ids) + answer_ids + [tokenizer.eos_token_id]
    
    if  len(input_ids)>17000:
        input_ids = question_ids + [tokenizer.eos_token_id]
        attention_mask = len(input_ids)*[1]
        labels = [-100]*len(input_ids) 
        print("exceeded limit - shud be 3 !")
        return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
        }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
        }

# Preprocessing function
def preprocess_function_val(examples):
    question = examples["input"]
    answer = examples["label"]
    
    messages = [
    {"role": "system", "content": "Please think step-by-step and give final answer using \boxed{}"},
    {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    question_ids = tokenizer(text, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    
    input_ids = question_ids + answer_ids + [tokenizer.eos_token_id]
    attention_mask = len(input_ids)*[1]
    labels = [-100]*len(question_ids) + answer_ids + [tokenizer.eos_token_id]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
        }


# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=train_dataset.column_names)
tokenized_val_dataset = eval_dataset.map(preprocess_function_val, batched=False, remove_columns=eval_dataset.column_names)

################ Set Training amd launch training ################
training_args = TrainingArguments(
    output_dir="./full_fft",
    report_to = "none",
    save_strategy="steps",
    evaluation_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    # eval_steps = 15,
    save_steps = 40,
    per_device_eval_batch_size= 1,
    num_train_epochs=1,
    # weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    # save_total_limit=2,
    # load_best_model_at_end=True,
    bf16=True,  # Use FP16 if GPU is available
    # dataloader_drop_last=False,
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    # eval_on_start = True
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
# trainer.model.print_trainable_parameters()
trainer.accelerator.print(f"{trainer.model}")
trainer.train()
