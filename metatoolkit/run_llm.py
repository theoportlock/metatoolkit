#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Run a local LLM with reproducible output.")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name or path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for deterministic)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling (1.0 disables)")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (non-deterministic)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt")

    generation_args = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
    }

    if args.top_k > 0:
        generation_args["top_k"] = args.top_k
    if args.top_p < 1.0:
        generation_args["top_p"] = args.top_p

    output = model.generate(**inputs, **generation_args)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

