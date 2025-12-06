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
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def get_model_and_tokenizer(model_path: str = "app/models/lora_output"):
    """
    Load model and tokenizer.
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
            raise FileNotFoundError(
                f"Model not found at '{model_path}' and base model not found at '{base_path}'. Please run 'make install'."
            )

    print(f"[INFO] Loading model from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return model, tokenizer


# Global cache for model/tokenizer to avoid reloading on every request in "dev" mode
# In a real prod app, you might manage this differently.
_MODEL = None
_TOKENIZER = None


def _ensure_model_loaded():
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _MODEL, _TOKENIZER = get_model_and_tokenizer()
    return _MODEL, _TOKENIZER


def generate_response(prompt: str) -> str:
    """
    Non-streaming generation (legacy).
    """
    try:
        model, tokenizer = _ensure_model_loaded()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("[INFO] Generating response...")
        outputs = model.generate(**inputs, max_new_tokens=4096)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"[ERROR] Failed to run inference: {e}"


def stream_response(prompt: str):
    """
    Generator that streams the response token by token.
    """
    try:
        model, tokenizer = _ensure_model_loaded()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=4096)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    except Exception as e:
        yield f"[ERROR] Failed to stream inference: {e}"


if __name__ == "__main__":
    # Test run
    print("Testing non-streaming:")
    print(generate_response("酒驾撞人怎么判刑？"))

    print("\nTesting streaming:")
    for chunk in stream_response("酒驾撞人怎么判刑？"):
        print(chunk, end="", flush=True)
    print()
