"""
app/comparison.py

Purpose:
    Provide side-by-side generation for the base model and the LoRA-adapted
    model so the frontend can compare their outputs in a single request.

Inputs:
    - prompt (str): user question.

Outputs:
    - NDJSON stream with entries shaped as:
        {"model": "base" | "lora", "delta": "<text chunk>", "done": bool}
"""

from __future__ import annotations

import json
import os
from threading import Thread
from typing import Generator, Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


BASE_MODEL_PATH = "app/models/base"
LORA_ADAPTER_PATH = "app/models/law-qa-qwen-lora"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

_SHARED_MODEL = None
_SHARED_TOKENIZER = None


def _load_shared_model():
    """
    Load and cache a single model + tokenizer instance.

    The returned model is a PeftModel with the LoRA adapter attached.
    We toggle the adapter on/off during generation to produce both base
    and LoRA outputs without loading two models into memory.
    """
    global _SHARED_MODEL, _SHARED_TOKENIZER
    if _SHARED_MODEL is None or _SHARED_TOKENIZER is None:
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(
                f"Base model not found at '{BASE_MODEL_PATH}'. Please run 'make install'."
            )
        if not os.path.exists(LORA_ADAPTER_PATH):
            raise FileNotFoundError(
                f"LoRA adapter not found at '{LORA_ADAPTER_PATH}'. Please place the adapters or rerun training."
            )

        print(
            f"[comparison] Loading shared model from '{BASE_MODEL_PATH}' with adapter '{LORA_ADAPTER_PATH}'..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        peft_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, trust_remote_code=True
        )

        _SHARED_MODEL = peft_model
        _SHARED_TOKENIZER = tokenizer

    return _SHARED_MODEL, _SHARED_TOKENIZER


def _generate_stream_part(
    *,
    prompt: str,
    label: str,
    model,
    tokenizer,
    use_adapter: bool,
) -> Iterable[str]:
    """
    Generate a stream for a single variant (base or LoRA) using the shared model.
    """
    try:
        # Use a system prompt to align with the training/intended usage
        messages = [
            {"role": "system", "content": "你是一个专业的法律咨询助手。"},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def _run_generate():
            generation_kwargs = dict(
                inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS
            )
            if use_adapter:
                model.generate(**generation_kwargs)
                return

            # Disable adapters for base output
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    model.generate(**generation_kwargs)
            else:
                model.generate(**generation_kwargs)

        thread = Thread(target=_run_generate)
        thread.start()

        for delta in streamer:
            yield (
                json.dumps(
                    {"model": label, "delta": delta, "done": False}, ensure_ascii=False
                )
                + "\n"
            )

        thread.join()
        yield (
            json.dumps({"model": label, "delta": "", "done": True}, ensure_ascii=False)
            + "\n"
        )

    except Exception as exc:  # noqa: BLE001
        yield (
            json.dumps(
                {
                    "model": label,
                    "delta": f"[ERROR] Failed to generate: {exc}",
                    "done": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def stream_compare(prompt: str) -> Generator[str, None, None]:
    """
    Stream responses for both base and LoRA models as NDJSON lines.
    """
    model, tokenizer = _load_shared_model()

    # Sequential generation to minimize memory footprint.
    for chunk in _generate_stream_part(
        prompt=prompt, label="base", model=model, tokenizer=tokenizer, use_adapter=False
    ):
        yield chunk

    for chunk in _generate_stream_part(
        prompt=prompt, label="lora", model=model, tokenizer=tokenizer, use_adapter=True
    ):
        yield chunk


def load_models():
    """
    Pre-load models into memory to avoid latency on the first request.
    """
    print("[comparison] Pre-loading shared model...")
    _load_shared_model()
    print("[comparison] All models loaded successfully.")
