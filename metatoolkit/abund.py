#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)

def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def abund(df, order=None, figsize=(4, 4), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    kwargs['ax'] = ax
    tdf = df.copy()
    if tdf.columns.shape[0] > 20:
        tdf['others'] = tdf[tdf.mean().sort_values(ascending=False).iloc[21:].index].sum(axis=1)
    tdf = norm(tdf.loc[:, tdf.mean().sort_values(ascending=False).iloc[:20].index])
    if order:
        tdf = tdf.loc[:, tdf.mean().sort_values(ascending=True).index]
    tdf.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    ax.set_ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def savefig(subject_path):
    subject_path = Path(subject_path)
    subject_name = subject_path.stem  # strips .tsv or any extension
    plt.tight_layout()
    plt.savefig(f'../results/{subject_name}_abundance.svg')

def parse_args():
    parser = argparse.ArgumentParser(description='''
    Abund - Produces an Abundance/Relative abundance plot of a given dataset
    ''')
    parser.add_argument('subject', help='Input file or subject name')
    parser.add_argument('--figsize', nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches, e.g., --figsize 4 4', default=[4, 4])
    return parser.parse_args()

def main():
    args = parse_args()
    subject = args.subject
    figsize = tuple(args.figsize)

    df = load(subject)
    abund(df, figsize=figsize)
    savefig(subject)

if __name__ == '__main__':
    main()

