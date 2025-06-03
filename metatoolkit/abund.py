#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

app = typer.Typer(help="Plot relative abundances from a TSV file.")


def plot_abundance(
    df: pd.DataFrame,
    order: bool,
    max_categories: int,
    figsize: tuple
):
    # Keep top N columns + combine others
    col_means = df.mean()
    sorted_cols = col_means.sort_values(ascending=False)

    if len(sorted_cols) > max_categories:
        top_cols = sorted_cols.iloc[:max_categories].index
        other_cols = sorted_cols.iloc[max_categories:].index
        df["others"] = df[other_cols].sum(axis=1)
        df = df[top_cols.tolist() + ['others']]

    # Normalize rows
    df = df.T.div(df.sum(axis=1), axis=1).T

    if order:
        df = df.loc[:, df.mean().sort_values().index]

    # Plot
    ax = df.plot(kind="bar", stacked=True, figsize=figsize, width=0.9, cmap="tab20")
    ax.set_ylabel("Relative abundance")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return plt


@app.command()
def main(
    input: Path = typer.Argument(..., help="TSV file with samples as rows, features as columns"),
    output: Path = typer.Option("abund.svg", "--output", "-o", help="Output file path"),
    figsize: List[float] = typer.Option([4, 4], help="Figure size, e.g. --figsize 6 5"),
    order: bool = typer.Option(False, help="Sort categories by average abundance"),
    max_categories: int = typer.Option(20, help="Number of top categories to keep before combining 'others'")
):
    """Plot stacked bar chart of relative abundances."""
    if not input.exists():
        typer.echo(f"❌ File not found: {input}")
        raise typer.Exit()

    if len(figsize) != 2:
        typer.echo("❌ --figsize must have two numbers: width and height")
        raise typer.Exit()

    df = pd.read_csv(input, sep="\t", index_col=0)
    plot = plot_abundance(df, order=order, max_categories=max_categories, figsize=tuple(figsize))
    plot.tight_layout()
    plot.savefig(output)
    typer.echo(f"✅ Saved plot to: {output}")


if __name__ == "__main__":
    app()
