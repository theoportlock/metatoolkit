#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify
import os
import itertools # Keep for potential future use, though not directly used in simplified interaction logic below

np.random.seed(42)
n = 1000

# 1. Load formula string from file
formula_file = "conf/formula.txt"
if not os.path.exists(formula_file):
    print(f"Error: Formula file '{formula_file}' not found.")
    print("Please create 'conf/formula.txt' with your desired formula (e.g., '2*x + 3*y + 1*x*y + noise').")
    exit(1)

with open(formula_file) as f:
    formula_str = f.read().strip()

# 2. Parse formula
expr = parse_expr(formula_str)
vars_in_formula = sorted({str(s) for s in expr.free_symbols})

# 3. Generate independent uniform(0,1) features for each variable except noise
features = {}
for v in vars_in_formula:
    if v.lower() == "noise":
        continue
    features[v] = np.random.uniform(0, 1, n)

df = pd.DataFrame(features)

# 4. Add noise term (normal 0, std=0.1)
noise_std = 0.1
if "noise" in vars_in_formula:
    df["noise"] = np.random.normal(0, noise_std, n)

# 5. Lambdify formula for fast evaluation
symbols_list = symbols(' '.join(vars_in_formula))
func = lambdify(symbols_list, expr, modules='numpy')

inputs = [df[var] for var in vars_in_formula]
df["y"] = func(*inputs)

# 6. Compute theoretical variance explained for terms
#    This section accurately calculates the variance of products of independent Uniform(0,1) variables.

def compute_variance_of_product_of_uniforms(factors):
    """
    Computes the variance of a product of independent U(0,1) variables.
    For k independent U(0,1) variables (X1, X2, ..., Xk):
    Var(X1*X2*...*Xk) = E[(X1*...*Xk)^2] - (E[X1*...*Xk])^2
                      = E[X1^2]*...*E[Xk^2] - (E[X1]*...*E[Xk])^2
                      = (1/3)^k - (1/4)^k
    """
    num_uniform_factors = 0
    for factor in factors:
        if isinstance(factor, sp.Symbol):
            num_uniform_factors += 1
        elif isinstance(factor, sp.Pow) and isinstance(factor.base, sp.Symbol) and isinstance(factor.exp, (sp.Integer, int)):
            num_uniform_factors += factor.exp
        # Constants are handled by the coefficient

    if num_uniform_factors == 0:
        return 0 # A product of only constants has zero variance

    return (1/3)**num_uniform_factors - (1/4)**num_uniform_factors

def compute_term_variance_contribution(term):
    """
    Computes the variance contribution of a single additive term from the formula.
    Assumes independent Uniform(0,1) variables for features.
    """
    coeff, factors = term.as_coeff_mul()
    coeff = float(coeff)

    if not factors: # If term is just a constant (e.g., '5')
        return 0

    variance_of_symbolic_product = compute_variance_of_product_of_uniforms(factors)
    return (coeff ** 2) * variance_of_symbolic_product


# Break down the expression into additive terms
terms = expr.as_ordered_terms()

# Store variances for direct and interaction effects
direct_variance_contributions = {} # For terms with only one feature (main effects)
all_interaction_terms_variance = {} # Stores (tuple_of_symbols: variance)

for term in terms:
    symbols_in_term = sorted(str(s) for s in term.free_symbols)

    if "noise" in symbols_in_term:
        continue # Noise is handled separately

    var_contribution = compute_term_variance_contribution(term)

    if len(symbols_in_term) == 1:
        feature_name = symbols_in_term[0]
        direct_variance_contributions[feature_name] = direct_variance_contributions.get(feature_name, 0) + var_contribution
    elif len(symbols_in_term) > 1:
        interaction_key = tuple(symbols_in_term)
        all_interaction_terms_variance[interaction_key] = all_interaction_terms_variance.get(interaction_key, 0) + var_contribution

# Calculate noise variance
noise_variance_contribution = noise_std**2 if "noise" in vars_in_formula else 0

# Calculate total variance for normalization
total_variance = sum(direct_variance_contributions.values()) + sum(all_interaction_terms_variance.values()) + noise_variance_contribution

if total_variance == 0:
    print("\nWarning: Total theoretical variance is zero. This can happen for a formula that's a constant (e.g., '5').")
    print("No variance percentages can be calculated.")
else:
    # 7. Breakdown and Output results

    # --- Feature Explained Variance (Inclusive of Interactions) ---
    feature_total_explained_variance = {}
    for feature in features.keys(): # Iterate over all features that exist in the dataframe
        # Start with the direct contribution of this feature
        feature_total_explained_variance[feature] = direct_variance_contributions.get(feature, 0)

        # Add its proportional share from all interactions it's part of
        for interaction_key, var_val in all_interaction_terms_variance.items():
            if feature in interaction_key:
                # Distribute the interaction variance equally among features involved
                feature_total_explained_variance[feature] += var_val / len(interaction_key)

    print("\nFeature Total Variance Explained (Inclusive of Interactions, %):")
    for k, v in sorted(feature_total_explained_variance.items()):
        print(f"  {k}: {100*v/total_variance:.2f}%")

    # --- Interaction Variance Explained (source, target, relative_importance) ---
    # This mimics SHAP by assigning half of the pairwise interaction variance to each feature
    # and for higher-order interactions, it assigns a proportional share.
    # The 'source, target' concept for variance is less direct than SHAP's attribution.
    # Here, we interpret 'source' as one feature in the interaction, and 'target' as another.
    # For a term like x*y*z, we enumerate all pairs (x,y), (x,z), (y,z) within it.

    interaction_breakdown_results = []

    # First, handle main effects for the 'source, target, RI' format (source=target)
    for feature, var_val in direct_variance_contributions.items():
        if total_variance > 0:
            ri_percent = var_val / total_variance
            interaction_breakdown_results.append({"source": feature, "target": feature, "relative_importance": ri_percent})

    # Then, handle interaction effects
    for interaction_key, var_val in all_interaction_terms_variance.items():
        num_features_in_interaction = len(interaction_key)

        # This is a simplified distribution. For SHAP, the specific interaction values
        # are calculated. Here, we're distributing the *total variance* of the interaction term.
        # For simplicity and to fit "source, target", we'll consider all unique pairs within the interaction.
        # For an interaction with N features, there are N*(N-1)/2 unique pairs.

        # If it's a 2-way interaction (e.g., x*y)
        if num_features_in_interaction == 2:
            f1, f2 = interaction_key
            # Split the variance equally between (f1,f2) and (f2,f1) "interaction links"
            split_variance = var_val / 2
            if total_variance > 0:
                ri_percent = split_variance / total_variance
                interaction_breakdown_results.append({"source": f1, "target": f2, "relative_importance": ri_percent})
                interaction_breakdown_results.append({"source": f2, "target": f1, "relative_importance": ri_percent})
        elif num_features_in_interaction > 2:
            # For higher-order interactions, the "source, target" mapping becomes ambiguous
            # without a more complex Shapley-like decomposition.
            # We will list all pairwise combinations, and assign a small share.
            # A more rigorous approach would require a functional ANOVA decomposition that
            # explicitly calculates higher-order interaction effects and their variances.
            # For this simplified request, we'll assign an even smaller share to each pair.

            # Let's consider distributing the variance across all unique pairs within the interaction
            # This is a heuristic to fit the "source, target" format for N-way interactions.
            all_pairs = list(itertools.combinations(interaction_key, 2))
            if len(all_pairs) > 0:
                variance_per_pair = var_val / len(all_pairs) / 2 # Divide by 2 because we list (f1,f2) and (f2,f1)
                for f1, f2 in all_pairs:
                    if total_variance > 0:
                        ri_percent = variance_per_pair / total_variance
                        interaction_breakdown_results.append({"source": f1, "target": f2, "relative_importance": ri_percent})
                        interaction_breakdown_results.append({"source": f2, "target": f1, "relative_importance": ri_percent})
            else: # Should not happen if num_features_in_interaction > 1
                 pass # No pairs to list

    # Add noise contribution
    if "noise" in vars_in_formula and total_variance > 0:
        noise_ri_percent = noise_variance_contribution / total_variance
        interaction_breakdown_results.append({"source": "noise", "target": "noise", "relative_importance": noise_ri_percent})

    # Sort results for cleaner output
    interaction_breakdown_results_df = pd.DataFrame(interaction_breakdown_results)
    if not interaction_breakdown_results_df.empty:
        interaction_breakdown_results_df = interaction_breakdown_results_df.sort_values(
            by=["source", "target"]).reset_index(drop=True)

        print("\nInteraction Variance Explained (source, target, relative_importance")
        print(interaction_breakdown_results_df.to_string(index=False))
    else:
        print("\nNo interaction variance to display.")


# 8. Save dataset and variance results
df.to_csv("results/dataset.tsv", sep='\t') 

if total_variance > 0:
    # Save Feature Total Variance Explained
    feature_total_explained_df = pd.DataFrame([
        {"feature": k, "total_variance_explained_percent": v/total_variance}
        for k, v in feature_total_explained_variance.items()
    ])
    feature_total_explained_df.to_csv("results/feature_total_variance_inclusive.tsv", sep='\t', index=False)

    # Save Interaction Variance Breakdown
    if not interaction_breakdown_results_df.empty:
        interaction_breakdown_results_df.to_csv("results/interaction_variance_breakdown.tsv", sep='\t', index=False)
        print("\n✅ Toy dataset and theoretical variance explained results saved to results/")
    else:
        print("\nToy dataset saved to results/, but no interaction variance breakdown due to zero interactions or total variance.")
else:
    print("\nToy dataset saved to results/, but variance summary not saved due to zero total variance.")
