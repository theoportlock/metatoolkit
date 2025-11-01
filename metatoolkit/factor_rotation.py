#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors

def main():
    parser = argparse.ArgumentParser(
        description="Rotate factor loadings (e.g. varimax) using statsmodels.rotate_factors"
    )
    parser.add_argument("input", help="Input CSV/TSV file with loadings (rows: variables, cols: factors). First column optional index.")
    parser.add_argument("-o", "--output", help="Output file (default: stdout).")
    parser.add_argument("--sep", default="\t", help="Delimiter for input/output (default tab). Use ',' for CSV.")
    parser.add_argument("--method", default="varimax",
                        choices=["varimax", "quartimax", "oblimin", "promax", "equamax", "quartimax", "oblimin"],
                        help="Rotation method (default: varimax).")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to perform Kaiser (column) normalization before rotating (normalize=True).")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="Maximum number of iterations for rotation (default 1000).")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance for convergence (default 1e-6).")
    args = parser.parse_args()

    # Read loadings
    df = pd.read_csv(args.input, sep=args.sep, index_col=0)
    loadings = df.values  # shape (n_variables, n_factors)

    # statsmodels.rotate_factors expects shape (n_factors, n_variables)? Let me check:
    # Actually rotate_factors takes A as (n_vars, n_factors)
    # See signature: rotate_factors(A, method, …) → (rotated, T)  :contentReference[oaicite:1]{index=1}

    # Call rotate_factors
    rotated, rotation_matrix = rotate_factors(
        loadings,
        method=args.method,
        normalize=args.normalize,
        max_iter=args.maxiter,
        tol=args.tol
    )

    # rotated is same shape as loadings
    rotated_df = pd.DataFrame(rotated, index=df.index, columns=df.columns)
    # Optionally also output rotation matrix
    rotmat_df = pd.DataFrame(rotation_matrix, index=df.columns, columns=df.columns)

    # Output
    if args.output:
        # write two sheets (or two files) — here as two files
        out_rot = args.output
        out_rotmat = args.output + ".rotation_matrix"
        rotated_df.to_csv(out_rot, sep=args.sep)
        rotmat_df.to_csv(out_rotmat, sep=args.sep)
        print(f"Rotated loadings → {out_rot}")
        print(f"Rotation matrix → {out_rotmat}")
    else:
        # print both to stdout (separated)
        print("# Rotated loadings:")
        print(rotated_df.to_csv(sep=args.sep))
        print("# Rotation matrix:")
        print(rotmat_df.to_csv(sep=args.sep))


if __name__ == "__main__":
    main()

