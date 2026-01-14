#!/usr/bin/env python

import muon as mu
import os
import argparse

parser = argparse.ArgumentParser(description="Run MOFA+ on MuData object")
parser.add_argument('--input', default='results/mofa/mdata.h5mu')
parser.add_argument('--output', default='results/mofa/t1.hdf5')
parser.add_argument('--factors', type=int, default=15)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

print("Running MOFA+...")
mdata = mu.read(args.input)
mu.tl.mofa(
    mdata,
    use_obs='union', # Fills in the missing pieces
    n_factors=args.factors,
    convergence_mode='medium',
    gpu_mode=True,
    outfile=args.output
)
print(f"MOFA+ model saved to {args.output}")

