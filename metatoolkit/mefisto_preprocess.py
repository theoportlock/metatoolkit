#!/usr/bin/env python3
# mefisto_preprocess.py

import argparse
import pandas as pd
import os
from pathlib import Path
import numpy as np


def load_data(file_paths):
    data_dict = {}
    for file_path in file_paths:
        name = Path(file_path).stem
        df = pd.read_csv(file_path, sep="\t", index_col=0)
        data_dict[name] = df
    return data_dict


def harmonize_samples(data_dict, time_df):
    # Keep only samples that have time metadata and are shared across time-aware datasets
    time_samples = set(time_df.index)
    time_aware_views = {k: df for k, df in data_dict.items() if df.index.isin(time_samples).all()}
    common_samples = set.intersection(*(set(df.index) for df in time_aware_views.values()))
    harmonized_data = {}
    for k, df in data_dict.items():
        df_h = df.loc[df.index.intersection(common_samples)]
        harmonized_data[k] = df_h
    return harmonized_data, list(common_samples)


def build_metadata(common_samples, time_file):
    time_df = pd.read_csv(time_file, sep="\t", index_col=0)
    missing = set(common_samples) - set(time_df.index)
    if missing:
        raise ValueError(f"Missing time metadata for samples: {missing}")
    meta_df = time_df.loc[common_samples]
    return meta_df


def write_outputs(data_dict, metadata, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in data_dict.items():
        df.to_csv(outdir / f"{name}.tsv", sep="\t")
    metadata.to_csv(outdir / "metadata.tsv", sep="\t")


def main():
    parser = argparse.ArgumentParser(description="Preprocess multiomic datasets for MEFISTO.")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="List of .tsv files (samples x features). All will be written. Time-aware views will be aligned.")
    parser.add_argument("-t", "--time", required=True,
                        help="TSV file with time variable, indexed by sample IDs.")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Output directory for harmonized data.")
    args = parser.parse_args()

    data_dict = load_data(args.input)
    time_df = pd.read_csv(args.time, sep="\t", index_col=0)

    harmonized_data, common_samples = harmonize_samples(data_dict, time_df)
    metadata = build_metadata(common_samples, args.time)

    write_outputs(harmonized_data, metadata, args.outdir)


if __name__ == "__main__":
    main()

