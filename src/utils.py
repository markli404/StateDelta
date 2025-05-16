import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from src.models.hs_edit_models import EditQwen2ForCausalLM, EditLlamaForCausalLM


def load_only_generation_config(model_name_or_path):
    config_path = os.path.join(model_name_or_path, "generation_config.json")
    assert os.path.exists(config_path), f"generation config {config_path} not found."
    generation_config = json.load(open(config_path, "r"))
    return generation_config


def load_model(model_name_or_path, load_method):
    if load_method != "sde":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
    else:
        if "qwen" in model_name_or_path.lower():
            model = EditQwen2ForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        elif "llama" in model_name_or_path.lower():
            model = EditLlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        else:
            raise ValueError(f"model_name_or_path {model_name_or_path} not found in EditModel.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    return model, tokenizer, model_config