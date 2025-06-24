################ Imports ################

# pip install transformers==4.45 accelerate hf_transfer datasets pandas 
# pip install flash-attn --no-build-isolation

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd
from transformers import DataCollatorForSeq2Seq

################ Load Model and Tokenizer ################
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  
train_file = "10k_samples_trimmed.csv"
# Define the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype= torch.bfloat16, attn_implementation = "flash_attention_2", device_map = "auto")
LR = 2e-5 

# Ensure the tokenizer adds a padding token (important for batch processing)
if tokenizer.pad_token is None:
    print("fixing pad token")
    tokenizer.pad_token = tokenizer.eos_token
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

################ Load Dataset amd preprocess ################
ds = Dataset.from_pandas(pd.read_csv(train_file))
train_test_split = ds.train_test_split(test_size=0.03)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Preprocessing function
def preprocess_function(examples):
    question = examples["User"]
    answer = examples["Assistant"]+"</think>"
    
    messages = [
    {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    question_ids = tokenizer(text, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, max_length=20512,truncation = True, add_special_tokens=False)["input_ids"]
    
    input_ids = question_ids + answer_ids
    attention_mask = len(input_ids)*[1]
    labels = [-100]*len(question_ids) + answer_ids 
    
    # if  len(input_ids)>20000:
    #     input_ids = question_ids + [tokenizer.eos_token_id]
    #     attention_mask = len(input_ids)*[1]
    #     labels = [-100]*len(input_ids) 
    #     print("exceeded limit - shud be 3 !")
    #     return {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "labels": labels
    #     }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
        }


# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=train_dataset.column_names)
tokenized_val_dataset = eval_dataset.map(preprocess_function, batched=False, remove_columns=eval_dataset.column_names)

################ Set Training amd launch training ################
training_args = TrainingArguments(
    output_dir="./full_fft",
    report_to = "wandb",
    save_strategy="epoch",
    evaluation_strategy="no",
    learning_rate=LR,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 64,
    # eval_steps = 15,
    # save_steps = 40,
    # per_device_eval_batch_size= 1,
    num_train_epochs=1,
    # weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    # save_total_limit=2,
    # load_best_model_at_end=True,
    bf16=True,  # Use FP16 if GPU is available
    # dataloader_drop_last=False,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
)
do_padding = True if training_args.per_device_train_batch_size > 1 else False
# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding = do_padding), # Add padding = True if BS > 1
)
# trainer.model.print_trainable_parameters()
trainer.accelerator.print(f"{trainer.model}")
trainer.train()
