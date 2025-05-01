#!/usr/bin/env Rscript

library(ggplot2)
library(optparse)
library(tools)

option_list <- list(
  make_option(c("-c", "--column"), type="character", default="sig", 
              help="Column to plot [default = %default]", metavar="character")
)

parser <- OptionParser(usage = "%prog [options] subject", option_list=option_list)
args <- parse_args(parser, positional_arguments = TRUE)

if (length(args$args) == 0) {
  stop("Error: Subject file must be provided")
}

subject <- args$args[1]
column <- args$options$column

if (file.exists(subject)) {
  subject <- file_path_sans_ext(basename(subject))
}

file_path <- paste0("../results/", subject, ".tsv")
df <- read.csv(file_path, sep="\t", header=TRUE, row.names=1)

p <- ggplot(df, aes(x = .data[[column]])) +
  geom_bar() +
  theme_minimal()

output_path <- paste0("../results/", subject, "bar.svg")
ggsave(output_path, plot=p, width=3, height=3)
