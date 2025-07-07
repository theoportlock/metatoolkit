#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(anthro)
})

# Define options
option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "Input TSV file", metavar = "file"),
  make_option(c("-o", "--output"), type = "character", help = "Output TSV file [default: stdout]", default = ""),
  make_option("--sex", type = "character", default = "sex", help = "Column name for sex [default: %default]"),
  make_option("--age", type = "character", default = "age", help = "Column name for age [default: %default]"),
  make_option("--age_in_month", action = "store_true", default = FALSE, help = "Set if age is in months"),
  make_option("--weight", type = "character", default = "weight", help = "Column name for weight [default: %default]"),
  make_option("--lenhei", type = "character", default = "length", help = "Column name for length/height [default: %default]"),
  make_option("--measure", type = "character", default = "measure", help = "Column name for measure ('l' or 'h') [default: %default]"),
  make_option("--headc", type = "character", default = NULL, help = "Column name for head circumference"),
  make_option("--armc", type = "character", default = NULL, help = "Column name for arm circumference"),
  make_option("--triskin", type = "character", default = NULL, help = "Column name for triceps skinfold"),
  make_option("--subskin", type = "character", default = NULL, help = "Column name for subscapular skinfold"),
  make_option("--oedema", type = "character", default = NULL, help = "Column name for oedema status")
)

# Parse options
opt <- parse_args(OptionParser(option_list = option_list))

# Read TSV
data <- read_tsv(opt$input, col_types = cols())

# Extract required columns with safe defaults
get_col <- function(colname, default = NA_real_) {
  if (!is.null(colname) && colname %in% colnames(data)) {
    data[[colname]]
  } else {
    rep(default, nrow(data))
  }
}

# Apply the anthro_zscores function
zscores <- anthro_zscores(
  sex = data[[opt$sex]],
  age = data[[opt$age]],
  is_age_in_month = opt$age_in_month,
  weight = get_col(opt$weight),
  lenhei = get_col(opt$lenhei),
  measure = get_col(opt$measure, NA_character_),
  headc = get_col(opt$headc),
  armc = get_col(opt$armc),
  triskin = get_col(opt$triskin),
  subskin = get_col(opt$subskin),
  oedema = get_col(opt$oedema, NA_character_)
)

# Combine with input
output <- bind_cols(data, zscores)

# Write result
if (opt$output == "") {
  write_tsv(output, file = "")
} else {
  write_tsv(output, file = opt$output)
}

