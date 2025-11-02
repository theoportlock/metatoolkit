#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter beta diversity values to within-subject distances from a specified baseline timepoint (includes baseline=0.0)."
    )
    parser.add_argument("beta", help="Path to beta diversity file (TSV with columns: source, target, bray-curtis)")
    parser.add_argument("meta", help="Path to metadata file (TSV with columns: sampleID, subjectID, timepoint)")
    parser.add_argument("-m", "--metric", default="bray-curtis",
                        help="Column name of the distance metric (default: bray-curtis)")
    parser.add_argument("-b", "--baseline", type=float, default=0.0,
                        help="Timepoint value to use as baseline (default: 0.0)")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file path")
    return parser.parse_args()

def main():
    args = parse_arguments()

    beta = pd.read_csv(args.beta, sep="\t")
    meta = pd.read_csv(args.meta, sep="\t")

    # Ensure required columns exist
    for col in ["source", "target", args.metric]:
        if col not in beta.columns:
            raise ValueError(f"Missing required column '{col}' in beta file")
    for col in ["sampleID", "subjectID", "timepoint"]:
        if col not in meta.columns:
            raise ValueError(f"Missing required column '{col}' in metadata file")

    meta_dict = meta.set_index("sampleID").to_dict(orient="index")
    results = []

    # For each subject, find baseline sample(s)
    subjects = meta["subjectID"].unique()
    for subj in subjects:
        subj_samples = meta[meta["subjectID"] == subj]
        baseline_samples = subj_samples[subj_samples["timepoint"] == args.baseline]["sampleID"].tolist()

        if not baseline_samples:
            continue  # no baseline for this subject

        # Add baseline itself with distance 0.0
        for b in baseline_samples:
            results.append({"sampleID": b, f"{args.metric}_{args.baseline}": 0.0})

        # Compare all non-baseline samples to baseline
        for _, row in beta.iterrows():
            s1, s2, dist = row["source"], row["target"], row[args.metric]
            if s1 not in meta_dict or s2 not in meta_dict:
                continue

            subj1, subj2 = meta_dict[s1]["subjectID"], meta_dict[s2]["subjectID"]
            if subj1 != subj2 or subj1 != subj:
                continue

            t1, t2 = meta_dict[s1]["timepoint"], meta_dict[s2]["timepoint"]
            if t1 == args.baseline and t2 != args.baseline:
                results.append({"sampleID": s2, f"{args.metric}_{args.baseline}": dist})
            elif t2 == args.baseline and t1 != args.baseline:
                results.append({"sampleID": s1, f"{args.metric}_{args.baseline}": dist})

    out_df = pd.DataFrame(results).drop_duplicates("sampleID")
    out_df.to_csv(args.output, sep="\t", index=False)
    print(f"✅ Saved {len(out_df)} within-subject baseline distances (baseline={args.baseline}) to: {args.output}")

if __name__ == "__main__":
    main()

