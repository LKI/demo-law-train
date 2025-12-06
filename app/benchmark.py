"""
app/benchmark.py

Purpose:
    Run benchmark on the model using a subset of the training/test data.
    Evaluates using Rouge-L score.
"""

import os
import json
import argparse
import datasets
from tqdm import tqdm
from rouge_score import rouge_scorer
import torch

from app.inference import get_model_and_tokenizer

# Configuration
DEFAULT_LIMIT = 50
OUTPUT_FILE = "benchmark_results.jsonl"
DATA_DIR = "app/data/train"  # Matches download.py


def load_local_dataset(data_dir):
    """
    Load dataset from local directory. attempt to find json/jsonl/parquet files.
    """
    try:
        # Explicitly look for jsonl files to avoid ambiguity
        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".jsonl")
        ]
        if files:
            print(
                f"[INFO] Found {len(files)} jsonl files: {[os.path.basename(f) for f in files]}"
            )
            return datasets.load_dataset("json", data_files=files, split="train")
        else:
            # Fallback to default if no jsonl (maybe parquet?)
            return datasets.load_dataset("parquet", data_dir=data_dir, split="train")
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        raise e


def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run benchmark on the SFT model")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model_path", type=str, default="app/models/lora_output", help="Path to model"
    )
    args = parser.parse_args()

    print(f"[INFO] Starting benchmark with limit={args.limit}...")

    # Load Model
    model, tokenizer = get_model_and_tokenizer(args.model_path)

    # Load Dataset
    print(f"[INFO] Loading dataset from {DATA_DIR}...")
    try:
        ds = load_local_dataset(DATA_DIR)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        # Fallback to downloading if empty (though make install should have done it)
        # For now, we assume it exists or we fail.
        return

    # Shuffle and select subset
    if len(ds) > args.limit:
        ds = ds.shuffle(seed=42).select(range(args.limit))

    results = []
    predictions = []
    references = []

    print("[INFO] Generating responses...")
    for item in tqdm(ds):
        # DISC-Law-SFT format usually has 'input' and 'output' or 'instruction'
        # Adjust based on actual columns. Common: 'input', 'output'
        # We will inspect the item keys if needed, but assuming input/output for now.
        system_prompt = "你是一个法律助手。"
        user_input = item.get("input", item.get("instruction", ""))
        ground_truth = item.get("output", "")

        if not user_input:
            continue

        # Format prompt using chat template if available
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7, do_sample=True
            )

        # Extract response (slice off input prompt)
        # Simple slicing: len(inputs.input_ids[0])
        response_ids = outputs[0][len(inputs.input_ids[0]) :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        results.append(
            {
                "input": user_input,
                "prediction": response_text,
                "ground_truth": ground_truth,
            }
        )
        predictions.append(response_text)
        references.append(ground_truth)

    # Compute Score
    print("[INFO] Computing Rouge-L...")
    avg_rouge = compute_metrics(predictions, references)

    print("\n[RESULT] Benchmark Complete.")
    print(f"[RESULT] Average Rouge-L: {avg_rouge:.4f}")

    # Save Results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"[INFO] Detailed results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
