#!/usr/bin/env python
import argparse
import pandas as pd
from pathlib import Path

# -------------------------
# Custom Aggregation Helpers
# -------------------------
# Defining these as named functions ensures the output columns
# use "q25" instead of "<lambda>".
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)
def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
def n_nonzero(x): return (x != 0).sum()
def n_total(x): return len(x)
def prev(x): return (x != 0).mean()

AGG_MAP = {
    "mean": "mean", "median": "median", "sum": "sum",
    "std": "std", "first": "first", "last": "last",
    "q25": q25, "q75": q75, "iqr": iqr,
    "n_nonzero": n_nonzero, "n_total": n_total, "prev": prev
}

def group(df, group_by=None, funcs=None):
    if funcs is None: funcs = ["sum"]

    # Expand "prev" into the three-metric suite if requested
    selected_funcs = []
    for f in funcs:
        if f == "prev":
            selected_funcs.extend(["n_nonzero", "n_total", "prev"])
        elif f in AGG_MAP:
            selected_funcs.append(f)
        else:
            raise ValueError(f"Unsupported aggregation function: {f}")

    # Unify Grouping: Use a dummy series to group "all" samples
    is_all = not group_by or (isinstance(group_by, list) and group_by[0].lower() == "all")
    group_key = [lambda _: "all"] if is_all else group_by

    # Aggregate
    # Note: Pandas agg is NaN-aware by default for mean/std/sum
    res = df.groupby(group_key).agg({col: [AGG_MAP[f] for f in selected_funcs] for col in df.columns})

    # Robust Column Flattening
    # This handles MultiIndex levels and preserves the "Feature_Stat" format
    res.columns = [f"{feat}_{stat}" for feat, stat in res.columns]

    if is_all:
        res.index = ["all"]
        res.index.name = None

    return res

# -------------------------
# Validation & I/O
# -------------------------

def merge_meta(df, meta_path, group_by):
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)

    # Restore Metadata Validation
    if group_by and not (len(group_by) == 1 and group_by[0].lower() == "all"):
        missing = [c for c in group_by if c not in meta.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")
        df = df.join(meta[group_by], how="inner")
    return df

def load_data(path_or_name):
    path = Path(path_or_name)
    if not path.is_file():
        path = Path("results") / f"{path_or_name}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path_or_name}")
    return pd.read_csv(path, sep="\t", index_col=0)

def main():
    parser = argparse.ArgumentParser(description="Group and aggregate datasets.")
    parser.add_argument("subject")
    parser.add_argument("--group_by", nargs="+")
    parser.add_argument("--func", nargs="+", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--meta")
    args = parser.parse_args()

    df = load_data(args.subject)
    if args.meta:
        df = merge_meta(df, args.meta, args.group_by)

    out = group(df, group_by=args.group_by, funcs=args.func)

    # Save logic
    func_str = "_".join(args.func)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t")
    print(out)

if __name__ == "__main__":
    main()
