#!/usr/bin/env python

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def parse_args():
    parser = argparse.ArgumentParser(description="Generic Kaplan-Meier Survival Analysis CLI")

    # Data arguments
    parser.add_argument("--input", required=True, help="Path to primary data file (TSV/CSV)")
    parser.add_argument("--meta", required=True, help="Path to metadata file (TSV/CSV)")
    parser.add_argument("--index", default="sampleID", help="Column to join files on")
    parser.add_argument("--subject", default="subjectID", help="Column identifying unique subjects")

    # Analysis arguments
    parser.add_argument("--time", required=True, help="Column for time/duration (e.g., timepoint)")
    parser.add_argument("--event", required=True, help="Column for the event (1=event, 0=censored)")
    parser.add_argument("--group", help="Column to stratify by (e.g., Condition)")

    # Plotting & Output
    parser.add_argument("--output", help="Output file path (e.g., results/plot.svg)")
    parser.add_argument("--no-ci", action="store_true", help="Disable confidence intervals")
    parser.add_argument("--figsize", default="3,3", help="Figure size as width,height (default=3,3)")

    return parser.parse_args()


def process_survival_data(df, args):
    def summarize_subject(group):
        event_rows = group[group[args.event] == 1]

        if not event_rows.empty:
            duration = event_rows[args.time].min()
            event_observed = 1
        else:
            duration = group[args.time].max()
            event_observed = 0

        res = {"duration": duration, "event": event_observed}

        if args.group:
            res["group_val"] = group[args.group].iloc[0]

        return pd.Series(res)

    return df.groupby(args.subject).apply(summarize_subject, include_groups=False).reset_index()


def remove_ci_outlines(ax):
    """Remove edges from CI shaded regions."""
    for poly in ax.collections:
        poly.set_edgecolor("none")
        poly.set_linewidth(0)


def main():
    args = parse_args()

    # Load and Merge
    sep_in = '\t' if args.input.endswith('.tsv') else ','
    sep_meta = '\t' if args.meta.endswith('.meta') or args.meta.endswith('.tsv') else ','

    df_in = pd.read_csv(args.input, sep=sep_in)
    df_meta = pd.read_csv(args.meta, sep=sep_meta)

    full_df = pd.merge(df_in, df_meta, on=args.index)

    # Process survival structure
    surv_df = process_survival_data(full_df, args)

    # Plotting
    width, height = map(float, args.figsize.split(','))
    fig, ax = plt.subplots(figsize=(width, height))

    # --- Spine & Tick Customization ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(0.4)
    ax.spines['bottom'].set_linewidth(0.4)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.tick_params(width=0.4)
    # ----------------------------------

    kmf = KaplanMeierFitter()

    plot_kwargs = {
        "ax": ax,
        "ci_show": not args.no_ci,
        "show_censors": True,
        "linewidth": 0.4,
        "censor_styles": {"ms": 3, "mew": 0.4},
        "ci_alpha": 0.25
    }

    if args.group:

        unique_groups = surv_df["group_val"].dropna().unique()

        if len(unique_groups) == 2:
            g1, g2 = unique_groups
            mask = surv_df["group_val"] == g1

            results = logrank_test(
                surv_df[mask]["duration"],
                surv_df[~mask]["duration"],
                surv_df[mask]["event"],
                surv_df[~mask]["event"]
            )

            print(f"Log-Rank Test ({g1} vs {g2}): p-value = {results.p_value:.4f}")

        for g in unique_groups:
            mask = surv_df["group_val"] == g

            kmf.fit(
                surv_df[mask]["duration"],
                surv_df[mask]["event"],
                label=str(g)
            )

            kmf.plot_survival_function(**plot_kwargs)

        # Remove CI outlines
        remove_ci_outlines(ax)

    else:

        kmf.fit(
            surv_df["duration"],
            surv_df["event"],
            label="Total Population"
        )

        kmf.plot_survival_function(**plot_kwargs)

        remove_ci_outlines(ax)

    plt.xlabel(f"Time ({args.time})")
    plt.ylabel("Probability")
    plt.grid(False)

    if args.output:
        out_dir = os.path.dirname(args.output)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.savefig(args.output, bbox_inches="tight")
        print(f"File saved: {args.output}")

    else:
        plt.show()


if __name__ == "__main__":
    main()
