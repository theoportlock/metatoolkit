#!/usr/bin/env python

import pandas as pd
import scanpy as sc
import muon as mu
from pathlib import Path
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser(description="Load modalities and metadata into MuData object")
parser.add_argument('--t1-dir', default='results/timepoints/yr1',
                    help='Directory containing T1 (baseline) data files')
#parser.add_argument('--t2-dir', default='results/timepoints/yr2',
#                    help='Directory containing T2 (after refeeding) data files')
#parser.add_argument('--notp-dir', default='results/timepoints/notp',
#                    help='Directory containing data with no timepoint info (optional)')
parser.add_argument('--meta-file', default='results/filtered/meta.tsv')
parser.add_argument('--output', default='results/mofa/mdata.h5mu')
args = parser.parse_args()

Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

print("Loading metadata...")
obs = pd.read_csv(args.meta_file, sep='\t', index_col=0)

print("Loading modalities for all timepoints...")
mods = {}

def load_timepoint_data(tp_name, tp_dir):
    """Helper to load all data files from a given directory into AnnData objects."""
    data_files = glob(f'{tp_dir}/*')
    if not data_files:
        print(f"Warning: No data files found in {tp_dir}")
        return {}
    tp_mods = {}
    for f in data_files:
        view_name = f"{Path(f).stem}_{tp_name}"  # keep timepoint in view name if needed
        adata = sc.AnnData(pd.read_csv(f, sep='\t', index_col=0))
        adata.obs['timepoint'] = tp_name
        tp_mods[view_name] = adata
    return tp_mods

mods.update(load_timepoint_data('t1', args.t1_dir))
#mods.update(load_timepoint_data('t2', args.t2_dir))
#mods.update(load_timepoint_data('notp', args.notp_dir))

# Merge into MuData
mdata = mu.MuData(mods)

# Join metadata
mdata.obs = mdata.obs.join(obs, how='inner')

print("Saving MuData object...")
mdata.write(args.output)
print(f"Saved MuData to {args.output}")

