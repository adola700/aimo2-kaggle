import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]= "1"
import torch
from huggingface_hub import HfFolder

# Save your token locally
HfFolder.save_token("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")

from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "cpu",torch_dtype = torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_id)
print("tokenizer length:", len(tok))
for param in model.parameters():
    param.data = param[:len(tok)]
    break

for name, param in model.named_parameters():
    if name == "lm_head.weight":
        param.data = param[:len(tok)]
        print(param.shape)

model_save_path = "akh99/" + model_id.split("/")[1] + "-tok-norm"
model.config.vocab_size = len(tok)
model.save_pretrained(model_save_path)
tok.save_pretrained(model_save_path)