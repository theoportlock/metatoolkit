#!/usr/bin/env Rscript

library(ggplot2)
library(optparse)
library(tools)

option_list <- list(
  make_option(c("-c", "--column"), type="character", default="sig", 
              help="Column to plot [default = %default]", metavar="character"),
  make_option(c("-b", "--bins"), type="integer", default=30, 
              help="Number of bins for the histogram [default = %default]", metavar="integer")
)

parser <- OptionParser(usage = "%prog [options] subject", option_list=option_list)
args <- parse_args(parser, positional_arguments = TRUE)

if (length(args$args) == 0) {
  stop("Error: Subject file must be provided")
}

subject <- args$args[1]
column <- args$options$column
bins <- args$options$bins

if (file.exists(subject)) {
  subject <- file_path_sans_ext(basename(subject))
}

file_path <- paste0("results/", subject, ".tsv")
df <- read.csv(file_path, sep="\t", header=TRUE, row.names=1)

p <- ggplot(df, aes(x = .data[[column]])) +
  geom_histogram(bins = bins, fill="blue", color="black", alpha=0.7) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1))

output_path <- paste0("results/", subject, "hist.svg")
ggsave(output_path, plot=p, width=8, height=6)
