#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols


def parse_args():
    parser = argparse.ArgumentParser(description="Display symbolic formula from file using SymPy.")
    parser.add_argument("--formula_file", type=str, default="formula.txt",
                        help="Path to the file containing the formula string (default: formula.txt)")
    parser.add_argument("--latex", action="store_true",
                        help="Display the formula as LaTeX")
    parser.add_argument("--save_latex", type=str, default=None,
                        help="Optional path to save LaTeX output (e.g. formula.tex)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load formula string
    with open(args.formula_file, "r") as f:
        formula_str = f.read()

    # Replace any known logical expressions (custom preprocessing)
    formula_str = formula_str.replace("(TimeOfDay == 'Afternoon') & (Promotion == 1)", "AfternoonPromo")

    # Declare symbols used in the formula
    Temperature_C, MarketingSpend_USD, AfternoonPromo, noise = symbols(
        'Temperature_C MarketingSpend_USD AfternoonPromo noise'
    )

    # Parse symbolic expression
    expr = parse_expr(formula_str)

    # Pretty-print expression
    print("\n📘 Parsed symbolic formula:\n")
    sp.pprint(expr, use_unicode=True)

    if args.latex or args.save_latex:
        latex_expr = sp.latex(expr)
        print("\n🧮 LaTeX representation:\n")
        print(latex_expr)

        if args.save_latex:
            with open(args.save_latex, "w") as f:
                f.write(latex_expr + "\n")
            print(f"\n✅ LaTeX saved to {args.save_latex}")


if __name__ == "__main__":
    main()

