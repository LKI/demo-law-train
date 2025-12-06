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


def generate_response(prompt: str, model_path: str = "app/models/lora_output") -> str:
    """
    Mock generation.
    TODO: Load model and tokenizer, generate text.
    """
    return f"[MOCK REACT] I received your question: '{prompt}'. As an AI Law assistant, I would advise checking the Civil Code..."


if __name__ == "__main__":
    # Test run
    print(generate_response("How do I sue for breach of contract?"))
