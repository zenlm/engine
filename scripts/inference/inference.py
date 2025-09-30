#!/usr/bin/env python3
"""
Inference script for MODEL_NAME
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with MODEL_NAME")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model", type=str, default="zenlm/MODEL_NAME", help="Model path")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{text}")


if __name__ == "__main__":
    main()
