#!/usr/bin/env python
# ==============================================================================
# mefisto_run.py
# This script runs the MEFISTO model on a preprocessed AnnData object and saves
# the trained model.
# ==============================================================================
import argparse
import mofapy2
from mofapy2.run.entry_point import entry_point
import anndata as ad
import os
import numpy as np

def run_mefisto_model(input_h5ad, n_factors, output_model, time_col, remove_zero_variance):
    """
    Loads an AnnData object, runs the MEFISTO model with a temporal
    component, and saves the trained model.

    Args:
        input_h5ad (str): Path to the preprocessed AnnData file.
        n_factors (int): The number of factors to use in the MEFISTO model.
        output_model (str): Path to save the trained MEFISTO model (.hdf5).
        time_col (str): The column in adata.obs containing timepoint information.
        remove_zero_variance (bool): If True, removes features with zero variance.
    """
    try:
        if not os.path.exists(input_h5ad):
            raise FileNotFoundError(f"Input file not found: {input_h5ad}")

        # Load the AnnData object
        print("Loading data...")
        try:
            adata = ad.read_h5ad(input_h5ad)
            print("Loaded an AnnData object.")
        except Exception as e:
            raise IOError(f"Failed to load AnnData object: {e}")

        if time_col not in adata.obs.columns:
            raise ValueError(f"Time column '{time_col}' not found in metadata.")

        # Optional: remove zero variance features
        if remove_zero_variance:
            print("Removing zero variance features...")
            if adata.shape[1] > 0:
                X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                nonzero_var_mask = np.std(X_data, axis=0) != 0
                if np.sum(nonzero_var_mask) < adata.shape[1]:
                    print(f"  - Removing {adata.shape[1] - np.sum(nonzero_var_mask)} / {adata.shape[1]} features.")
                    adata = adata[:, nonzero_var_mask]
                else:
                    print("  - No zero-variance features found.")

        # Extract time data and ensure it's in the correct format (string)
        print("Extracting time data...")
        time_data = adata.obs[time_col].astype(str).values

        # Configure and build the MEFISTO model
        print("Configuring MEFISTO model...")
        mofa = entry_point()

        # Use AnnData object directly
        print("Setting data options and loading from AnnData object...")
        mofa.set_data_options(
            scale_groups=False,
            scale_views=False
        )
        mofa.set_data_from_anndata(adata)

        mofa.set_model_options(factors=n_factors)

        # Add the time covariate and set interpolation options
        print("Setting covariates and interpolation options...")
        mofa.set_covariates(time_data, "time")
        # Extract unique timepoints and convert to float for interpolation knots
        unique_timepoints = np.sort(adata.obs[time_col].unique()).astype(float)
        mofa.set_interpolation_options(
            interpolation_mode="linear",
            interpolation_knots=unique_timepoints
        )

        mofa.set_train_options(iter=100)

        print("Building the model...")
        mofa.build()

        print("Running the model...")
        mofa.run()

        # Save the model
        print(f"Saving trained model to '{output_model}'...")
        mofa.save(output_model)
        print(f"Trained MEFISTO model saved to '{output_model}'")

    except Exception as e:
        print(f"An error occurred while running the MEFISTO model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MEFISTO model on a preprocessed AnnData object with a temporal component."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the preprocessed AnnData file (.h5ad) from mefisto_preprocess."
    )
    parser.add_argument(
        "--n_factors",
        type=int,
        default=10,
        help="Number of factors for the MEFISTO model (default: 10)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output MEFISTO model file (.hdf5)."
    )
    parser.add_argument(
        "--time_col",
        required=True,
        help="Column in the metadata containing timepoint information (e.g., 'timepoint')."
    )
    parser.add_argument(
        "--remove_zero_variance",
        action='store_true',
        help="If set, removes features with zero variance before training."
    )
    args = parser.parse_args()

    run_mefisto_model(args.input, args.n_factors, args.output, args.time_col, args.remove_zero_variance)