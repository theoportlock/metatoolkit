#!/usr/bin/env python
# ==============================================================================
# mefisto_visualize.py
# This script loads a trained MEFISTO model and generates basic plots,
# saving them as image files.
# ==============================================================================
import argparse
import mofapy2
import matplotlib.pyplot as plt
import os
import anndata as ad
import numpy as np

def visualize_results(input_model, input_h5ad, output_dir, time_col):
    """
    Loads a trained MEFISTO model and generates basic visualization plots.

    Args:
        input_model (str): Path to the trained MEFISTO model file.
        input_h5ad (str): Path to the preprocessed AnnData file used for training.
        output_dir (str): Directory to save the output plots.
        time_col (str): The column in adata.obs containing timepoint information.
    """
    try:
        if not os.path.exists(input_model):
            raise FileNotFoundError(f"Input model file not found: {input_model}")
        if not os.path.exists(input_h5ad):
            raise FileNotFoundError(f"Input AnnData file not found: {input_h5ad}")

        # Load the trained model and the data
        model = mofapy2.mofa_model(input_model)
        adata = ad.read_h5ad(input_h5ad)
        print("Loaded an AnnData object for visualization.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print("Generating plots...")

        # Plot 1: Factor weights
        # Since we have a single AnnData object, we expect a single view in the model.
        # The default view name in mofapy2 is 'view_0' when loading from AnnData.
        if 'view_0' in model.get_weights():
            weights = model.get_weights()['view_0']
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap="viridis", aspect="auto")
            plt.title("Factor Weights")
            plt.xlabel("Factors")
            plt.ylabel("Features")
            plt.colorbar(label="Weight")
            plt.savefig(os.path.join(output_dir, "factor_weights.png"))
            print(f"Plot saved: {os.path.join(output_dir, 'factor_weights.png')}")
            plt.close()
        else:
            print("Could not find weights for 'view_0'. Available views: ", list(model.get_weights().keys()))


        # Plot 2: Variance explained
        var_expl = model.get_variance_explained()
        if 'view_0' in var_expl:
            view_var_expl = var_expl['view_0']
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(view_var_expl)), view_var_expl)
            plt.title("Variance Explained per Factor")
            plt.xlabel("Factor")
            plt.ylabel("Variance Explained")
            plt.xticks(range(len(view_var_expl)), [f"Factor {i+1}" for i in range(len(view_var_expl))])
            plt.savefig(os.path.join(output_dir, "variance_explained.png"))
            print(f"Plot saved: {os.path.join(output_dir, 'variance_explained.png')}")
            plt.close()
        else:
            print("Could not find variance explained for 'view_0'. Available views: ", list(var_expl.keys()))


        # Plot 3: Temporal Factors over Time
        if time_col not in adata.obs.columns:
            raise KeyError(f"The '{time_col}' column is missing from adata.obs.")

        unique_timepoints = np.sort(adata.obs[time_col].unique())
        
        # The factors are retrieved for the single view, which is the default.
        factors = model.get_factors()

        plt.figure(figsize=(12, 8))
        for factor_idx in range(factors.shape[1]):
            factor_values = factors[:, factor_idx]
            plt.plot(unique_timepoints, factor_values, marker='o', label=f'Factor {factor_idx + 1}')

        plt.title("Temporal Factors over Time")
        plt.xlabel("Timepoint")
        plt.ylabel("Factor Value")
        plt.legend(title="Factors")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temporal_factors.png"))
        print(f"Plot saved: {os.path.join(output_dir, 'temporal_factors.png')}")
        plt.close()


    except Exception as e:
        print(f"An error occurred during visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize results from a trained MEFISTO model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the trained MEFISTO model file (.hdf5) from mefisto_run."
    )
    parser.add_argument(
        "--input_h5ad",
        required=True,
        help="Path to the preprocessed AnnData file (.h5ad) used for training."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the output plots."
    )
    parser.add_argument(
        "--time_col",
        required=True,
        help="Column in the metadata containing timepoint information (e.g., 'timepoint')."
    )
    args = parser.parse_args()

    visualize_results(args.input, args.input_h5ad, args.output_dir, args.time_col)