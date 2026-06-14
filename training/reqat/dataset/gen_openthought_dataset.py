from datasets import load_dataset, DatasetDict, concatenate_datasets
import datasets
import transformers
import os
from transformers import AddedToken
import re
import random

tokenizer = transformers.AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')
eos = tokenizer.eos_token_id

def has_boxed(example):
    return bool(re.search(r"\\?boxed\{[^}]*\}", example["conversations"][1]["value"]))

def convert_to_token(example):
    user_text = tokenizer.decode(151644) # User
    assistant_text = tokenizer.decode(151645) # Assistant
    # text
    prompt = user_text+example['conversations'][0]['value']+assistant_text
    response = example['conversations'][1]['value']
    input_ids = tokenizer.encode(prompt+response)
    input_ids.append(eos)
    # mask
    mask = [1 for i in range(len(input_ids))]
    for j in range(len(tokenizer.encode(prompt))): mask[j] = 0
    new_conversations = {'input_ids': input_ids, 'completion_mask': mask}
    return new_conversations

# OpenThought3 with boxed
raw_train = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
raw_train = raw_train.filter(has_boxed) # 325K
raw_train = raw_train.filter(lambda x: x["domain"] == "math") # 850K = 53K * 16
raw_train = raw_train.shuffle(seed=42).select(range(int(89000//16)))
converted_train = raw_train.map(
    convert_to_token,
    remove_columns=raw_train.column_names
)

converted = DatasetDict({"train": converted_train})
output_dir = "OpenThought3-DeepSeek-89k-math-sft"
converted.save_to_disk(output_dir)

print(f"Saved with only train split to {output_dir}")
