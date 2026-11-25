#!/usr/bin/env python
# -*- coding: utf-8 -*-
# change.py

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


ALL_ANALYSES = ["mww", "fc", "diffmean", "smd", "summary"]


# ---------------- CLI ---------------- #

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Change - Bivariate analysis of feature changes"
    )

    parser.add_argument(
        "subject",
        help="Path to dataset file or subject name (resolved as results/{name}.tsv)"
    )

    parser.add_argument(
        "-df2", "--df2",
        required=True,
        help="Path to metadata/one-hot-encoded file or subject name"
    )

    parser.add_argument(
        "-c", "--columns",
        nargs="+",
        help="Metadata columns to split by (default: all)"
    )

    parser.add_argument(
        "-a", "--analysis",
        nargs="+",
        choices=ALL_ANALYSES,
        default=ALL_ANALYSES,
        help="Methods of analysis to perform"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output TSV file (default: results/{subject}_change.tsv)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed DataFrame output"
    )

    return parser.parse_args()


# ---------------- IO ---------------- #

def load_table(path_or_name: str) -> Tuple[pd.DataFrame, str]:
    """Load a TSV from file OR results/{name}.tsv."""
    p = Path(path_or_name)
    if p.is_file():
        stem = p.stem
        df = pd.read_csv(p, sep="\t", index_col=0)
    else:
        stem = path_or_name
        df = pd.read_csv(Path("results") / f"{stem}.tsv", sep="\t", index_col=0)
    return df, stem


def save(df: pd.DataFrame, outfile: Path, index: bool = True):
    """Save df to the specified output path."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=index)
    print(f"[OK] Saved results to {outfile}")


# ---------------- Operations ---------------- #

def splitter(df: pd.DataFrame, df2: pd.DataFrame, column: str) -> dict:
    """Split df rows based on df2[column] levels."""
    output = {}
    for level in df2[column].unique():
        idx = df2[df2[column] == level].index
        idx_common = idx.intersection(df.index)
        if len(idx_common) == 0:
            print(f"[WARNING] No overlapping indices for {column}='{level}'")
            continue
        output[level] = df.loc[idx_common].copy()
    return output


def mww(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    pvals = [mannwhitneyu(df1[col], df2[col]).pvalue for col in df1.columns]
    out = pd.DataFrame(pvals, index=df1.columns, columns=["MWW_pval"])
    out["MWW_qval"] = fdrcorrection(out["MWW_pval"])[1]
    return out


def fc(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    mean1 = df1.mean()
    mean2 = df2.mean()
    fold = mean1 / mean2
    out = pd.DataFrame(fold, columns=["FC"])
    out["Log2FC"] = np.log2(out["FC"])
    out["Log10FC"] = np.log10(out["FC"])
    return out


def diffmean(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    diff = df1.mean() - df2.mean()
    return pd.DataFrame(diff, columns=["diffmean"])


def smd(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference (Cohen's d) using the pooled SD.

    SMD = (mean1 - mean2) / sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    Features with zero (or undefined) pooled SD return NaN.
    """
    n1, n2 = len(df1), len(df2)
    dof = n1 + n2 - 2

    if dof <= 0:
        return pd.DataFrame(np.nan, index=df1.columns, columns=["SMD"])

    pooled_var = ((n1 - 1) * df1.var(ddof=1) + (n2 - 1) * df2.var(ddof=1)) / dof
    pooled_sd = np.sqrt(pooled_var).replace(0, np.nan)

    d = (df1.mean() - df2.mean()) / pooled_sd
    return pd.DataFrame(d, columns=["SMD"])


def summary(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    def stat_str(df: pd.DataFrame):
        m = df.mean().round(2).astype(str)
        s = df.std().round(2).astype(str)
        return m + " ± " + s

    return pd.DataFrame({
        "group1_summary": stat_str(df1),
        "group2_summary": stat_str(df2)
    })


# ---------------- Main Analysis ---------------- #

def change_analysis(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    columns: Optional[List[str]] = None,
    analyses: Optional[List[str]] = None
) -> pd.DataFrame:

    if columns is None:
        columns = meta.columns.tolist()
    if analyses is None:
        analyses = list(ALL_ANALYSES)

    methods = {
        "mww": mww,
        "fc": fc,
        "diffmean": diffmean,
        "smd": smd,
        "summary": summary,
    }
    results = []

    for col in columns:
        sub = splitter(df, meta, col)
        levels = list(sub.keys())

        for i, lvl1 in enumerate(levels):
            for lvl2 in levels[i + 1:]:
                df1, df2 = sub[lvl1], sub[lvl2]

                if df1.empty or df2.empty:
                    print(f"[WARNING] Skipping empty {col}: {lvl1} vs {lvl2}")
                    continue

                pieces = [methods[m](df1, df2) for m in analyses]
                combined = pd.concat(pieces, axis=1)
                combined.index.name = "feature"

                combined = combined.assign(
                    source=col,
                    comparison=f"{lvl1}_vs_{lvl2}"
                ).set_index(["source", "comparison"], append=True)

                results.append(combined)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results)
    out = out.reorder_levels(["source", "comparison", "feature"])
    return out


# ---------------- Entry Point ---------------- #

def main():
    args = parse_arguments()

    df, stem1 = load_table(args.subject)
    meta, stem2 = load_table(args.df2)

    out = change_analysis(df, meta, columns=args.columns, analyses=args.analysis)

    if not args.quiet:
        print(out)

    # Determine output path
    if args.output:
        outfile = Path(args.output)
    else:
        outfile = Path("results") / f"{stem1}_change.tsv"

    save(out, outfile)


if __name__ == "__main__":
    main()
