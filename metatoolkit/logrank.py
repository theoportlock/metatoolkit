#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
from lifelines.statistics import multivariate_logrank_test


def parse_args():
    p = argparse.ArgumentParser(description="Log-rank test CLI")

    # input
    p.add_argument("--input", required=True,
                   help="Primary data file (tsv/csv)")
    p.add_argument("--meta", required=True,
                   help="Metadata file (tsv/csv)")

    p.add_argument("--index", default="sampleID",
                   help="Column used to merge tables")

    p.add_argument("--subject", default="subjectID",
                   help="Subject identifier column")

    # survival variables
    p.add_argument("--time", required=True,
                   help="Time column (numeric)")

    p.add_argument("--event", required=True,
                   help="Event column (1=event, 0=censored)")

    p.add_argument("--group", required=True,
                   help="Grouping column")

    p.add_argument("--output",
                   help="Output TSV file (default stdout)")

    return p.parse_args()


def read_table(path):
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def collapse_survival(df, subject, time, event, group):

    rows = []

    for sid, g in df.groupby(subject):

        g = g.sort_values(time)

        # first event
        ev = g[g[event] == 1]

        if len(ev) > 0:
            duration = ev.iloc[0][time]
            event_observed = 1
        else:
            duration = g[time].max()
            event_observed = 0

        group_vals = g[group].dropna().unique()

        if len(group_vals) != 1:
            raise ValueError(
                f"{sid} has multiple group assignments: {group_vals}"
            )

        rows.append({
            "subject": sid,
            "duration": duration,
            "event": event_observed,
            "group": group_vals[0]
        })

    return pd.DataFrame(rows)


def main():

    args = parse_args()

    print("[metatoolkit] log-rank test", file=sys.stderr)

    # load data
    df = pd.merge(
        read_table(args.input),
        read_table(args.meta),
        on=args.index
    )

    # sanity checks
    for col in [args.subject, args.time, args.event, args.group]:
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found")

    if not set(df[args.event].dropna().unique()).issubset({0, 1}):
        sys.exit("ERROR: event column must contain 0/1 values")

    # collapse longitudinal data
    surv = collapse_survival(
        df,
        args.subject,
        args.time,
        args.event,
        args.group
    )

    print(f"Subjects: {len(surv)}", file=sys.stderr)
    print(f"Groups: {surv['group'].nunique()}", file=sys.stderr)

    # run log-rank
    res = multivariate_logrank_test(
        surv["duration"],
        surv["group"],
        surv["event"]
    )

    out = pd.DataFrame({
        "test": ["logrank"],
        "test_statistic": [res.test_statistic],
        "p_value": [res.p_value],
        "df": [res.degrees_of_freedom],
        "n_subjects": [len(surv)],
        "n_groups": [surv["group"].nunique()]
    })

    # output
    if args.output:
        out.to_csv(args.output, sep="\t", index=False)
    else:
        out.to_csv(sys.stdout, sep="\t", index=False)


if __name__ == "__main__":
    main()
