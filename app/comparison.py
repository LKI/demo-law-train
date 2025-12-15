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
from queue import Queue
from threading import Thread
from typing import Generator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


BASE_MODEL_PATH = "app/models/base"
LORA_ADAPTER_PATH = "app/models/law-qa-qwen-lora"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

_BASE_MODEL = None
_BASE_TOKENIZER = None
_LORA_MODEL = None
_LORA_TOKENIZER = None


def _load_base():
    """Load and cache the base model."""
    global _BASE_MODEL, _BASE_TOKENIZER
    if _BASE_MODEL is None or _BASE_TOKENIZER is None:
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(
                f"Base model not found at '{BASE_MODEL_PATH}'. Please run 'make install'."
            )
        print(f"[comparison] Loading base model from '{BASE_MODEL_PATH}'...")
        _BASE_TOKENIZER = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, trust_remote_code=True
        )
        _BASE_MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        _BASE_MODEL.eval()
    return _BASE_MODEL, _BASE_TOKENIZER


def _load_lora():
    """Load and cache the LoRA-adapted model."""
    global _LORA_MODEL, _LORA_TOKENIZER
    if _LORA_MODEL is None or _LORA_TOKENIZER is None:
        if not os.path.exists(LORA_ADAPTER_PATH):
            raise FileNotFoundError(
                f"LoRA adapter not found at '{LORA_ADAPTER_PATH}'. Please place the adapters or rerun training."
            )
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(
                f"Base model not found at '{BASE_MODEL_PATH}'. Please run 'make install'."
            )

        print(
            f"[comparison] Loading LoRA model using base '{BASE_MODEL_PATH}' and adapter '{LORA_ADAPTER_PATH}'..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        _LORA_MODEL = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        _LORA_MODEL.eval()
        _LORA_TOKENIZER = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, trust_remote_code=True
        )
    return _LORA_MODEL, _LORA_TOKENIZER


def _enqueue_stream(
    prompt: str,
    label: str,
    model,
    tokenizer,
    queue: Queue,
):
    """
    Run generation for one model and push deltas into a shared queue.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for delta in streamer:
            queue.put({"model": label, "delta": delta, "done": False})

        thread.join()
        queue.put({"model": label, "delta": "", "done": True})

    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "model": label,
                "delta": f"[ERROR] Failed to generate: {exc}",
                "done": True,
            }
        )


def stream_compare(prompt: str) -> Generator[str, None, None]:
    """
    Stream responses for both base and LoRA models as NDJSON lines.
    """
    base_model, base_tokenizer = _load_base()
    lora_model, lora_tokenizer = _load_lora()

    queue: Queue = Queue()

    # Start parallel generation
    workers = [
        Thread(
            target=_enqueue_stream,
            kwargs={
                "prompt": prompt,
                "label": "base",
                "model": base_model,
                "tokenizer": base_tokenizer,
                "queue": queue,
            },
            daemon=True,
        ),
        Thread(
            target=_enqueue_stream,
            kwargs={
                "prompt": prompt,
                "label": "lora",
                "model": lora_model,
                "tokenizer": lora_tokenizer,
                "queue": queue,
            },
            daemon=True,
        ),
    ]

    for worker in workers:
        worker.start()

    finished = 0
    total_models = len(workers)

    while finished < total_models:
        item = queue.get()
        if item.get("done"):
            finished += 1
        yield json.dumps(item, ensure_ascii=False) + "\n"

    for worker in workers:
        worker.join()


def load_models():
    """
    Pre-load models into memory to avoid latency on the first request.
    """
    print("[comparison] Pre-loading base model...")
    _load_base()
    print("[comparison] Pre-loading LoRA model...")
    _load_lora()
    print("[comparison] All models loaded successfully.")
