# ==============================================================================
# mefisto_run.py
# This script runs the MEFISTO model on the preprocessed data and saves
# the trained model object.
# ==============================================================================
import argparse
import mofapy2
import anndata as ad
import os
import sys

def run_mefisto_model(input_h5ad, n_factors, output_model):
    """
    Loads an AnnData object, runs the MEFISTO model, and saves the trained model.

    Args:
        input_h5ad (str): Path to the preprocessed AnnData file.
        n_factors (int): The number of factors to use in the MEFISTO model.
        output_model (str): Path to save the trained MEFISTO model (.hdf5).
    """
    try:
        if not os.path.exists(input_h5ad):
            raise FileNotFoundError(f"Input file not found: {input_h5ad}")

        # Load the AnnData object
        adata = ad.read_h5ad(input_h5ad)

        # Configure and build the MEFISTO model
        model = mofapy2.mofa_model()
        model.set_data_options(adata.X, "data_view")
        model.set_model_options(factors=n_factors, likelihoods=["gaussian"])
        model.set_train_options(iter=100) # Set a reasonable number of iterations
        model.build()

        # Run the model
        model.run()

        # Save the trained model
        model.save(output_model)
        print(f"Trained MEFISTO model saved to '{output_model}'")

    except Exception as e:
        print(f"An error occurred while running the MEFISTO model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MEFISTO model on preprocessed AnnData."
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
    args = parser.parse_args()
    
    run_mefisto_model(args.input, args.n_factors, args.output)
