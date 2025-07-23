#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import os
import itertools


def parse_args():
    parser = argparse.ArgumentParser(description="Compute expected variance contribution from formula terms.")
    parser.add_argument('--input', type=str, default='conf/formula.txt', help='Path to input formula file')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory to save TSV files')
    return parser.parse_args()


def compute_variance_of_product_of_uniforms(factors):
    k = 0
    for factor in factors:
        if isinstance(factor, sp.Symbol):
            k += 1
        elif isinstance(factor, sp.Pow) and isinstance(factor.base, sp.Symbol) and isinstance(factor.exp, (sp.Integer, int)):
            k += factor.exp
    return (1/3)**k - (1/4)**k if k > 0 else 0


def compute_term_variance_contribution(term):
    coeff, factors = term.as_coeff_mul()
    return float(coeff)**2 * compute_variance_of_product_of_uniforms(factors)


def compute_theoretical_variance(formula_str):
    expr = parse_expr(formula_str)
    all_terms = expr.as_ordered_terms()

    direct_variance = {}
    interaction_variance = {}

    for term in all_terms:
        features_in_term = sorted(str(s) for s in term.free_symbols)
        var_contribution = compute_term_variance_contribution(term)
        if len(features_in_term) == 1:
            key = features_in_term[0]
            direct_variance[key] = direct_variance.get(key, 0) + var_contribution
        elif len(features_in_term) > 1:
            key = tuple(features_in_term)
            interaction_variance[key] = interaction_variance.get(key, 0) + var_contribution

    total_variance = sum(direct_variance.values()) + sum(interaction_variance.values())

    if total_variance == 0:
        print("⚠️ Warning: total variance is zero. Skipping output.")
        return None, None

    # --- Feature-level (isolated) contributions ---
    feature_df = pd.DataFrame([
        {
            "feature": f,
            "importance": direct_variance.get(f, 0),
            "relative_importance": direct_variance.get(f, 0) / total_variance
        }
        for f in sorted(set(direct_variance.keys()).union(*interaction_variance.keys()))
    ])

    # --- Unified table: main effects + single listing of interaction effects ---
    interaction_rows = []

    # Main effects (isolated)
    for feature, var_val in direct_variance.items():
        interaction_rows.append({
            "source": feature,
            "target": feature,
            "importance": var_val,
            "relative_importance": var_val / total_variance
        })

    # Interaction terms (combinations only, not permutations)
    for interaction_key, var_val in interaction_variance.items():
        if len(interaction_key) == 2:
            f1, f2 = sorted(interaction_key)
            interaction_rows.append({
                "source": f1,
                "target": f2,
                "importance": var_val,
                "relative_importance": var_val / total_variance
            })
        elif len(interaction_key) > 2:
            # For higher-order terms like x*y*z, distribute across unique unordered pairs
            all_pairs = [tuple(sorted(p)) for p in itertools.combinations(interaction_key, 2)]
            var_per_pair = var_val / len(all_pairs)
            added_pairs = set()
            for f1, f2 in all_pairs:
                if (f1, f2) not in added_pairs:
                    interaction_rows.append({
                        "source": f1,
                        "target": f2,
                        "importance": var_per_pair,
                        "relative_importance": var_per_pair / total_variance
                    })
                    added_pairs.add((f1, f2))

    interaction_df = pd.DataFrame(interaction_rows).sort_values(["source", "target"]).reset_index(drop=True)
    return feature_df, interaction_df


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Formula file '{args.input}' not found.")
        exit(1)

    with open(args.input) as f:
        formula_str = f.read().strip()

    feature_df, interaction_df = compute_theoretical_variance(formula_str)

    if feature_df is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        feature_path = os.path.join(args.output_dir, 'feature_importance.tsv')
        interaction_path = os.path.join(args.output_dir, 'interaction_importance.tsv')

        feature_df.to_csv(feature_path, sep='\t', index=False)
        interaction_df.to_csv(interaction_path, sep='\t', index=False)

        print(f"✅ Saved feature importance to {feature_path}")
        print(f"✅ Saved interaction importance to {interaction_path}")


if __name__ == '__main__':
    main()

