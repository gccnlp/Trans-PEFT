import torch
import transformers
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn as nn
import numpy as np
import copy
from safetensors.torch import load_file
import os


def load_peft_and_merge(base_model_name_or_path, peft_id):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True
    )
    merged_model = PeftModel.from_pretrained(model,peft_id).merge_and_unload()
    merged_model_path = peft_id+"/on_"+base_model_name_or_path.split("/")[-1]
    
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    del merged_model
    del tokenizer
    return merged_model_path
