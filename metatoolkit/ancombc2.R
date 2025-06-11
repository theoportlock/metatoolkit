#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(data.table)
  library(ANCOMBC)
})

# Argument parser
parser <- ArgumentParser(description = "Run ANCOM-BC2 differential abundance with fixed and random effects.")
parser$add_argument("--abundance", required=TRUE, help="Path to abundance table (samples x species, TSV)")
parser$add_argument("--metadata", required=TRUE, help="Path to metadata table (TSV, index = sampleID)")
parser$add_argument("--formula", required=TRUE, help="Fixed effects formula, e.g. 'intervention * timepoint + sex'")
parser$add_argument("--rand_formula", required=FALSE, help="Random effects formula, e.g. '1 | subjectID'")
parser$add_argument("--group_col", required=TRUE, help="Grouping column for pairwise comparisons")
parser$add_argument("--output", required=TRUE, help="Path to output TSV file")

args <- parser$parse_args()

# Load abundance and metadata tables
otu_data <- read.csv(args$abundance, sep="\t", header=TRUE, row.names=1)
meta_data <- read.csv(args$metadata, sep="\t", header=TRUE, row.names=1)

# Align sample IDs
common_samples <- intersect(rownames(otu_data), rownames(meta_data))
otu_data <- otu_data[common_samples, , drop = FALSE]
meta_data <- meta_data[common_samples, , drop = FALSE]

# Warn if anything is dropped
if (length(common_samples) < nrow(otu_data) || length(common_samples) < nrow(meta_data)) {
  warning("Some samples were dropped due to mismatch between metadata and abundance tables.")
}

# Run ANCOM-BC2
set.seed(123)
output <- ancombc2(
  data = t(otu_data),
  meta_data = meta_data,
  fix_formula = args$formula,
  rand_formula = args$rand_formula,
  group = args$group_col,
  p_adj_method = "holm",
  pseudo_sens = TRUE,
  prv_cut = 0.10,
  lib_cut = 1000,
  s0_perc = 0.05,
  struc_zero = FALSE,
  neg_lb = FALSE,
  alpha = 0.05,
  n_cl = 2,
  verbose = TRUE,
  global = FALSE,
  pairwise = TRUE,
  dunnet = FALSE,
  trend = FALSE,
  iter_control = list(tol = 1e-5, max_iter = 20, verbose = FALSE),
  em_control = list(tol = 1e-5, max_iter = 100),
  lme_control = NULL,
  mdfdr_control = list(fwer_ctrl_method = "holm", B = 100),
  trend_control = NULL
)

# Extract and combine pairwise results
res_pair <- output$res_pair

if (length(res_pair) > 0) {
  result_df <- rbindlist(lapply(names(res_pair), function(term) {
    df <- as.data.frame(res_pair[[term]])
    df$comparison <- term
    df$feature <- rownames(df)
    df
  }), use.names = TRUE, fill = TRUE)

  fwrite(result_df, file = args$output, sep = "\t")
} else {
  cat("No pairwise results were returned by ANCOM-BC2.\n")
}

