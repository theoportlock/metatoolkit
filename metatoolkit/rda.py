#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.stats.ordination import rda
from sklearn.preprocessing import StandardScaler
import os

# Argument parser
parser = argparse.ArgumentParser(description="Perform RDA using scikit-bio")
parser.add_argument("-t", "--taxonomic_data", required=True, help="Path to taxonomic data file (TSV)")
parser.add_argument("-m", "--metadata", required=True, help="Path to metadata file (TSV)")
parser.add_argument("-f", "--fixed_effects", required=True,
                    help="Comma-separated list of fixed effects (e.g. 'Factor1,Factor2')")
parser.add_argument("-o", "--output", default="../results/lda_results.tsv", help="Path to output TSV file")

args = parser.parse_args()

# Load data
taxo = pd.read_csv(args.taxonomic_data, sep="\t", index_col=0)
metadata = pd.read_csv(args.metadata, sep="\t", index_col=0)

# Align sample IDs
common_samples = taxo.index.intersection(metadata.index)
taxo = taxo.loc[common_samples]
metadata = metadata.loc[common_samples]

# Select fixed effects
effects = args.fixed_effects.split(',')
X = metadata[effects].copy()

# Optionally standardize predictors
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables

# Prepare matrices for skbio rda
Y = taxo
X = X.loc[Y.index]

# Run RDA
result = rda(Y.values, X.values, scale_Y=False, scaling=2, sample_ids=Y.index.tolist())

# Save summary table
summary_df = pd.DataFrame({
    "Eigenvalues": result.eigvals,
    "Explained variance (%)": result.proportion_explained * 100
})
summary_df.to_csv(args.output, sep="\t")

# Plot ordination (scaling 1: distances between samples)
os.makedirs("../results", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 5))
sample_coords = result.samples.iloc[:, :2]
sns.scatterplot(x=sample_coords.iloc[:, 0], y=sample_coords.iloc[:, 1], ax=ax)
for i, sample in enumerate(sample_coords.index):
    ax.text(sample_coords.iloc[i, 0], sample_coords.iloc[i, 1], sample, fontsize=6)
ax.set_title("RDA ordination (Scaling 2)")
plt.savefig("../results/ordiplot1.svg")

# Print stats
print("Adjusted R²:", result.proportion_explained.sum())

