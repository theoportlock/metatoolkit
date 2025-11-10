#!/usr/bin/env python

import mofax as mofa
import muon as mu
import pandas as pd
import os
import argparse

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Export MOFA+ factors and loadings")
parser.add_argument('--model', default='results/mofa/t1.hdf5')
parser.add_argument('--mdata', default='results/mofa/mdata.h5mu')
parser.add_argument('--outdir', default='results/mofa')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- Load model and mdata ---
model = mofa.mofa_model(args.model)
mdata = mu.read(args.mdata)

# --- Export factors ---
factors_arr = model.get_factors()
sample_index = model.get_samples()

# Check shape match
if len(sample_index) != factors_arr.shape[0]:
    print(f"Error: Sample count mismatch. {len(sample_index)} sample names vs {factors_arr.shape[0]} factor rows.")
    exit()

# Extract just the 'sample' column if sample_index is a DataFrame
if isinstance(sample_index, pd.DataFrame):
    sample_index = sample_index.iloc[:, 1]  # or sample_index["sample"]

factors_df = pd.DataFrame(
       factors_arr,
       index=sample_index,
       columns=[f"Factor_{i+1}" for i in range(model.nfactors)]
       )

factors_df.to_csv(os.path.join(args.outdir, 'sample_factors.tsv'), sep='\t')
print("Exported sample factors.")

# --- Export loadings ---
weights = model.get_weights()
if not isinstance(weights, dict):
    weights = {model.get_views()[0]: weights}

for view_name, weights_arr in weights.items():
    try:
        features = mdata[view_name].var_names
    except KeyError:
        print(f"Skipping {view_name}, features not found.")
        continue

    if weights_arr.shape[0] != len(features):
        print(f"Skipping {view_name}, shape mismatch.")
        print(f"  weights_arr.shape[0] = {weights_arr.shape[0]}, len(features) = {len(features)}")
        continue

    df = pd.DataFrame(
        weights_arr,
        index=features,
        columns=[f"Factor_{i+1}" for i in range(model.nfactors)]
    )
    df.to_csv(os.path.join(args.outdir, f'loadings_{view_name}.tsv'), sep='\t')
    print(f"Exported loadings for {view_name}")

