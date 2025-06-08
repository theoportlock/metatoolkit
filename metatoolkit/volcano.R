#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(optparse)
  library(readr)
  library(dplyr)
  library(tools)
})

# Define CLI options
option_list <- list(
  make_option(c("-c", "--change"), type = "character", default = "coef", help = "Column for log2 fold change"),
  make_option(c("-s", "--sig"), type = "character", default = "qval", help = "Column for p/q-values"),
  make_option(c("--fc"), type = "double", default = 1.0, help = "Fold change threshold"),
  make_option(c("--pval"), type = "double", default = 0.05, help = "P-value threshold"),
  make_option(c("--no_annot"), action = "store_true", default = FALSE, help = "Disable annotation")
)

parser <- OptionParser(usage = "usage: %prog [options] input.tsv", option_list = option_list)
arguments <- parse_args(parser, positional_arguments = TRUE)

if (length(arguments$args) != 1) {
  stop("Exactly one input file must be provided.")
}

input_file <- arguments$args[1]
opts <- arguments$options

# Load data
df <- read_tsv(input_file, col_types = cols())
change_col <- opts$change
sig_col <- opts$sig
fc_thresh <- opts$fc
pval_thresh <- opts$pval
annotate <- !opts$no_annot

# Check required columns
if (!(change_col %in% colnames(df)) || !(sig_col %in% colnames(df))) {
  stop("Change or significance column not found in input data.")
}

# Prepare data
df <- df %>%
  mutate(log_pval = -log10(.data[[sig_col]]),
         significant = abs(.data[[change_col]]) > fc_thresh & .data[[sig_col]] < pval_thresh,
         label = ifelse(significant & annotate, rownames(df), NA))

# Plot
p <- ggplot(df, aes(x = .data[[change_col]], y = log_pval)) +
  geom_point(aes(color = significant), alpha = 0.5) +
  geom_vline(xintercept = c(-fc_thresh, 0, fc_thresh), color = c("red", "gray", "red"), linetype = c("solid", "dashed", "solid")) +
  geom_hline(yintercept = -log10(pval_thresh), color = "red", linetype = "solid") +
  scale_color_manual(values = c("black", "red")) +
  labs(x = "log2 fold change", y = "-log10 p-value") +
  theme_minimal()

if (annotate) {
  p <- p + geom_text_repel(aes(label = label), size = 2.5, max.overlaps = Inf)
}

# Save plot
outname <- paste0(file_path_sans_ext(basename(input_file)), "volcano.svg")
ggsave(outname, p, width = 8, height = 6)


