#!/usr/bin/env python

import mofax as mofa
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Visualize R² from MOFA+ model")
parser.add_argument('--model', default='results/mofa/t1.hdf5')
args = parser.parse_args()

breakpoint()
print("Loading MOFA model...")
model = mofa.mofa_model(args.model)
r2_long_df = model.get_variance_explained()

# Barplot
plt.figure(figsize=(4, 3))
sns.barplot(data=r2_long_df, x='View', y='R2', estimator=np.sum, errorbar=None)
plt.title('Total R² per View')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Heatmap
r2_df = r2_long_df.pivot_table(index='View', columns='Factor', values='R2')
plt.figure(figsize=(4, 3))
sns.heatmap(r2_df, cmap='viridis', vmax=15)
plt.title('R² per View and Factor')
plt.tight_layout()
plt.show()

