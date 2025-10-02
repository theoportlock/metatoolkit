# MetaToolkit

MetaToolkit is a collection of command-line and script-based tools for microbial community analysis, statistical profiling, visualization, and network construction. All scripts are standalone and can be sent and ran individually.

## Features

- 📊 Visualizations: Boxplots, barplots, volcano plots, upset plots, heatmaps, circos, PCoA, biplots, and more
- 🧮 Statistics: Kruskal-Wallis, chi-squared, Fisher’s exact, ANOVA, adonis, enterotyping, PCA, RDA, and mediation analysis
- 🧠 Interpretable ML: SHAP residuals, AUC/ROC, EV, and variance explanation utilities
- 🧬 Network and clustering: Graph generation, shortest path, Leiden clustering, tree plotting
- 🧰 Utilities: One-hot encoding, stratification, data merging, header formatting, and summary table generation

## Installation

Install via pip in editable mode:

```bash
pip install -e .
```

You will need Python ≥3.8 and packages listed in `requirements.txt`.

## Usage

Each script is modular and can be run independently. Common usage:

```bash
python metatoolkit/boxplot.py --input data.tsv --group GroupColumn --output fig.svg
```

You can also import functions for use in your own scripts:

```python
from metatoolkit import functions
```

## Example Scripts

| Script                     | Description                                     |
|---------------------------|-------------------------------------------------|
| `boxplot.py`              | Generate boxplots                               |
| `adonis.R`                | Run PERMANOVA using `vegan::adonis`             |
| `calculate_diversity.R`   | Compute alpha diversity metrics                 |
| `shap_residuals.py`       | Visualize SHAP residuals                        |
| `plot_network.py`         | Build and display network graphs                |
| `createsupptables.py`     | Create summary tables for publication           |
| `scale.py`                | Normalize or scale your dataset                 |
| `splitter.py`             | Train/test split with reproducibility           |

## Project Structure

```
metatoolkit/          # Python scripts and modules
tests/                # Unit tests
build/                # Build artifacts
setup.py              # Installation script
pyproject.toml        # Build configuration
requirements.txt      # Dependencies
LICENSE               # License file
README.md             # This file
```

## Development

To run the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.

