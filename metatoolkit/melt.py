#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


# ---------- Utility ---------- #

def log(msg):
    print(f"[melt] {msg}")

def load(path):
    return pd.read_csv(path, sep="\t")

def save(df, path, index=False):
    """Save DataFrame to a TSV file. Defaults to index=False for long-format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


# ---------- Melting Logic ---------- #

def apply_melt(df, **kw):
    """Melt DataFrame from wide to long format."""
    
    id_vars_arg = kw.get("id_vars")
    if not id_vars_arg:
        log("Error: Must specify --id_vars to melt.")
        return None

    id_vars = [c.strip() for c in id_vars_arg.split(",") if c.strip()]
    var_name = kw.get("var_name", "variable")
    value_name = kw.get("value_name", "value")

    # Validate id_vars exist
    missing_cols = [c for c in id_vars if c not in df.columns]
    if missing_cols:
        log(f"Error: Columns {missing_cols} not found in input.")
        return None

    before = df.shape
    try:
        df_melted = df.melt(
            id_vars=id_vars,
            var_name=var_name,
            value_name=value_name
        )
        log(f"Melt complete: {before} → {df_melted.shape}")
        return df_melted
    except Exception as e:
        log(f"Melt failed: {e}")
        return None


# ---------- CLI ---------- #

def parse_args():
    p = argparse.ArgumentParser(description="Melt a TSV file from wide to long format.")
    p.add_argument("input", help="Path to input TSV file")
    p.add_argument("-o", "--output", help="Path to output TSV file (default: adds _melt suffix)")
    
    p.add_argument("-i", "--id_vars", required=True, help="Column(s) to use as identifier variables (comma-separated)")
    p.add_argument("-vn", "--var_name", default="feature", help="Name for the new 'variable' column (default: feature)")
    p.add_argument("-val", "--value_name", default="value", help="Name for the new 'value' column (default: value)")
    
    return p.parse_args()


# ---------- Main ---------- #

def main():
    args = parse_args()

    df = load(args.input)

    out = apply_melt(df, **vars(args))

    if out is None:
        log("No output generated.")
        return

    output_path = args.output or f"{Path(args.input).stem}_melt.tsv"
    save(out, output_path)
    log(f"Saved melted output to: {output_path}")


if __name__ == "__main__":
    main()
