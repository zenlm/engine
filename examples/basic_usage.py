"""
Basic usage example for MODEL_NAME
"""
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Load model and tokenizer
    model_name = "zenlm/MODEL_NAME"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example prompts
    prompts = [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about AI."
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
