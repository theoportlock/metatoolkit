#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from kneed import KneeLocator

def main():
    parser = argparse.ArgumentParser(description="Scree (elbow) analysis from a saved PCA model.")
    parser.add_argument("model", help="Path to saved PCA model (joblib or pickle).")
    parser.add_argument("-o", "--output", help="Output TSV file with eigenvalues and explained variance ratios.")
    parser.add_argument("--sep", default="\t", help="Delimiter for TSV/CSV output (default: tab).")
    parser.add_argument("--plot", help="Optional scree plot output file (PNG/PDF).")
    args = parser.parse_args()

    # Load PCA model
    pca = joblib.load(args.model)

    # Extract explained variance and ratio
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_

    components = range(1, len(ev) + 1)
    outdf = pd.DataFrame({
        "Component": components,
        "Eigenvalue": ev,
        "ExplainedVarianceRatio": evr
    })

    # Elbow detection using KneeLocator
    kneedle = KneeLocator(
        components,
        ev,
        curve="convex",
        direction="decreasing"
    )
    elbow = kneedle.knee

    outdf["Elbow"] = ["<-- elbow" if c == elbow else "" for c in components]

    # Save table
    if args.output:
        outdf.to_csv(args.output, sep=args.sep, index=False)
    else:
        print(outdf.to_csv(sep=args.sep, index=False))

    # Plot
    if args.plot:
        plt.figure(figsize=(6,4))
        plt.plot(components, ev, marker="o")
        if elbow is not None:
            plt.axvline(elbow, color="red", linestyle="--", label=f"Elbow at {elbow}")
            plt.legend()
        plt.title("Scree Plot")
        plt.xlabel("Component")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=300)
        print(f"Scree plot saved to {args.plot}")

    if elbow:
        print(f"Optimal number of components (elbow): {elbow}")

if __name__ == "__main__":
    main()

