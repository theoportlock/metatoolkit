# ==============================================================================
# mefisto_visualize.py
# This script loads a trained MEFISTO model and generates basic plots,
# saving them as image files.
# ==============================================================================
import argparse
import mofapy2
import matplotlib.pyplot as plt
import os

def visualize_results(input_model, output_dir):
    """
    Loads a trained MEFISTO model and generates basic visualization plots.

    Args:
        input_model (str): Path to the trained MEFISTO model file.
        output_dir (str): Directory to save the output plots.
    """
    try:
        if not os.path.exists(input_model):
            raise FileNotFoundError(f"Input model file not found: {input_model}")
        
        # Load the trained model
        model = mofapy2.mofa_model(input_model)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print("Generating plots...")

        # Plot 1: Factor weights for the first view
        weights = model.get_weights()["data_view"]
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap="viridis", aspect="auto")
        plt.title("Factor Weights (data_view)")
        plt.xlabel("Factors")
        plt.ylabel("Features")
        plt.colorbar(label="Weight")
        plt.savefig(os.path.join(output_dir, "factor_weights.png"))
        print(f"Plot saved: {os.path.join(output_dir, 'factor_weights.png')}")
        plt.close()

        # Plot 2: Variance explained
        var_expl = model.get_variance_explained()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(var_expl["data_view"])), var_expl["data_view"])
        plt.title("Variance Explained per Factor")
        plt.xlabel("Factor")
        plt.ylabel("Variance Explained")
        plt.savefig(os.path.join(output_dir, "variance_explained.png"))
        print(f"Plot saved: {os.path.join(output_dir, 'variance_explained.png')}")
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
        "--output_dir",
        required=True,
        help="Directory to save the output plots."
    )
    args = parser.parse_args()
    
    visualize_results(args.input, args.output_dir)

