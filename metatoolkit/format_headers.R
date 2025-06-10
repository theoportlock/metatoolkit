#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript format_headers.R <input_file.tsv> <output_dir>")
}

input_file <- args[1]
output_dir <- args[2]

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Construct output filename
base <- tools::file_path_sans_ext(basename(input_file))
output_file <- file.path(output_dir, paste0(base, ".tsv"))

df <- read.table(input_file, sep="\t", header=TRUE, row.names=1)

# Write output
write.table(df, file=output_file, sep="\t", quote=FALSE)
