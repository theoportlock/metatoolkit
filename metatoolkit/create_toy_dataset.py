#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a toy dataset from a symbolic formula.")
    parser.add_argument('input', type=str, help='Path to formula file')
    parser.add_argument('--output', type=str, default='results/dataset.tsv', help='Path to output dataset file')
    parser.add_argument('--target', type=str, default='y', help='Name of target output column (default: y)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-n', type=int, default=1000, help='Number of samples to generate')
    return parser.parse_args()


def create_dataset(formula_str, n, seed, target):
    np.random.seed(seed)

    expr = parse_expr(formula_str)
    vars_in_formula = sorted({str(s) for s in expr.free_symbols})

    features = {
        v: np.random.uniform(-1, 1, n)
        for v in vars_in_formula
    }

    df = pd.DataFrame(features)

    func = lambdify(symbols(vars_in_formula), expr, modules='numpy')
    inputs = [df[var] for var in vars_in_formula]
    df[target] = func(*inputs)

    return df


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Formula file '{args.input}' not found.")
        exit(1)

    with open(args.input) as f:
        formula_str = f.read().strip()

    df = create_dataset(formula_str, args.n, args.seed, args.target)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep='\t')
    print(f"✅ Dataset saved to {args.output}")


if __name__ == '__main__':
    main()

