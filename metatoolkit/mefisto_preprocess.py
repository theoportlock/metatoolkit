#!/usr/bin/env python3
import argparse
import pandas as pd
import anndata as ad
import scanpy as sc
import os
import numpy as np

def preprocess_data(input_files, time_metadata_file, output_h5ad, skip_log_transform):
    """
    Loads multiple raw count data files and metadata, performs basic
    preprocessing, and saves the result as an AnnData object.

    Args:
        input_files (list): A list of paths to the raw count data CSV files.
        time_metadata_file (str): Path to the time metadata CSV file.
        output_h5ad (str): Path to save the preprocessed AnnData file.
        skip_log_transform (bool): If True, skips the log1p transformation.
    """
    try:
        # Load time metadata
        metadata = pd.read_csv(time_metadata_file, index_col=0, sep='\t')

        # Find intersection of samples across all datasets
        common_samples = set(metadata.index)
        for file_path in input_files:
            counts = pd.read_csv(file_path, index_col=0, sep='\t')
            common_samples &= set(counts.index)

        if not common_samples:
            raise ValueError("No common samples found across all datasets.")

        common_samples = list(common_samples)
        metadata = metadata.loc[common_samples]

        # Load and preprocess each data view
        adatas = []
        for i, file_path in enumerate(input_files):
            view_name = os.path.basename(file_path).split('.')[0]

            # Assuming the counts are tab-separated
            counts = pd.read_csv(file_path, index_col=0, sep='\t')

            # Ensure consistent sample order
            counts = counts.loc[common_samples]

            # Create AnnData object for the view
            adata = ad.AnnData(X=counts.values, obs=metadata.copy(), var=pd.DataFrame(index=counts.columns))
            adata.var['view'] = view_name

            # Basic preprocessing
            sc.pp.normalize_total(adata)

            if not skip_log_transform:
                sc.pp.log1p(adata)
            else:
                print(f"Skipping log1p transformation for view '{view_name}' as requested.")

            adatas.append(adata)

        # Concatenate AnnData objects if there are multiple views
        if len(adatas) > 1:
            # Make var names unique before concatenating
            for i, adata in enumerate(adatas):
                adata.var_names = adata.var['view'][0] + '_' + adata.var_names
            
            # Concatenate along features
            full_adata = ad.concat(adatas, axis=1, join='outer', label='view', uns_merge='unique')
        else:
            full_adata = adatas[0]


        # Save the AnnData object
        directory = os.path.dirname(output_h5ad)
        os.makedirs(directory, exist_ok=True)
        full_adata.write(output_h5ad)
        print(f"Preprocessed AnnData object saved to '{output_h5ad}'")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess multiple microbiome count datasets for MEFISTO analysis."
    )
    parser.add_argument(
        "-i", "--inputs",
        nargs='+', # Accepts one or more arguments
        required=True,
        help="Paths to the raw count data CSV files. Samples in columns, features in rows."
    )
    parser.add_argument(
        "-t", "--time_metadata",
        required=True,
        help="Path to the time metadata CSV file. Samples in rows, features in columns."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the output AnnData file (.h5ad)."
    )
    parser.add_argument(
        "--skip_log_transform",
        action='store_true', # No value needed, just presence of flag
        help="Skips the log1p transformation, useful for data with negative values (e.g., EEG)."
    )
    args = parser.parse_args()

    preprocess_data(args.inputs, args.time_metadata, args.output, args.skip_log_transform)