"""
app/inference.py

Purpose:
    Run inference/benchmarks on the trained model.
    Also provides a function to be called by the web API.

Inputs:
    - Model path (merged or base + adapter)
    - Prompt
Outputs:
    - Generated text
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(prompt: str, model_path: str = "app/models/lora_output") -> str:
    """
    Run inference using the model.
    Falls back to base model if the specified model_path does not exist.
    """
    # Check if model exists, fallback to base if not
    if not os.path.exists(model_path):
        base_path = "app/models/base"
        if os.path.exists(base_path):
            print(
                f"[INFO] Model not found at '{model_path}', falling back to base model at '{base_path}'"
            )
            model_path = base_path
        else:
            return f"[ERROR] Model not found at '{model_path}' and base model not found at '{base_path}'. Please run 'make install'."

    try:
        print(f"[INFO] Loading model from '{model_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Use float16 for efficiency, device_map='auto' to use GPU if available
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("[INFO] Generating response...")
        outputs = model.generate(**inputs, max_new_tokens=4096)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"[ERROR] Failed to run inference: {e}"


if __name__ == "__main__":
    # Test run
    print(generate_response("酒驾撞人怎么判刑？"))
