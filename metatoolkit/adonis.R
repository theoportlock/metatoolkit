#!/usr/bin/env Rscript

# adonis_generic.R
# Generic PERMANOVA (adonis2) runner for two-group comparisons
# Supports arbitrary metadata, formulas, distance metrics, and optional strata

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(vegan))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(tidyr))

# -----------------------------
# command line options
# -----------------------------
option_list <- list(
  make_option(c("--input"), type="character", help="Abundance TSV file."),
  make_option(c("--metadata"), type="character", help="Metadata TSV file."),
  make_option(c("--formula"), type="character", help="PERMANOVA formula (R syntax, RHS only)."),
  make_option(c("--distance"), type="character", default="bray", help="Distance metric (default: bray)."),
  make_option(c("--strata"), type="character", default=NULL, help="Optional strata column."),
  make_option(c("--timepoint_col"), type="character", default="timepoint", help="Timepoint column name."),
  make_option(c("--timepoint_a"), type="character", help="First timepoint value."),
  make_option(c("--timepoint_b"), type="character", help="Second timepoint value."),
  make_option(c("--sample_id_col"), type="character", default="sampleID", help="Sample ID column name."),
  make_option(c("--output"), type="character", default="permanova_results.tsv", help="Output TSV file.")
)

opt <- parse_args(OptionParser(option_list=option_list))

# -----------------------------
# required arguments
# -----------------------------
required <- c("input", "metadata", "formula", "timepoint_a", "timepoint_b")
missing <- required[sapply(required, function(x) is.null(opt[[x]]))]

if (length(missing) > 0) {
  stop(paste("Missing required arguments:", paste(missing, collapse=", ")), call.=FALSE)
}

# -----------------------------
# load data
# -----------------------------
abundance <- read_tsv(opt$input, show_col_types = FALSE)
metadata  <- read_tsv(opt$metadata, show_col_types = FALSE)

# -----------------------------
# set rownames
# -----------------------------
abundance_df <- abundance %>%
  column_to_rownames(opt$sample_id_col)

metadata_df <- metadata %>%
  column_to_rownames(opt$sample_id_col)

# -----------------------------
# align samples
# -----------------------------
common_samples <- intersect(rownames(abundance_df), rownames(metadata_df))

abundance_df <- abundance_df[common_samples, , drop = FALSE]
metadata_df  <- metadata_df[common_samples, , drop = FALSE]

# -----------------------------
# subset timepoints
# -----------------------------
if (!opt$timepoint_col %in% colnames(metadata_df)) {
  stop(paste("Timepoint column not found:", opt$timepoint_col), call.=FALSE)
}

metadata_filtered <- metadata_df %>%
  filter(.data[[opt$timepoint_col]] %in% c(opt$timepoint_a, opt$timepoint_b))

abundance_filtered <- abundance_df[rownames(metadata_filtered), , drop = FALSE]

metadata_filtered[[opt$timepoint_col]] <- droplevels(
  as.factor(metadata_filtered[[opt$timepoint_col]])
)

# -----------------------------
# infer variables used in model
# -----------------------------
formula_clean <- gsub("\\s+", "", opt$formula)
model_vars <- unlist(strsplit(formula_clean, "\\+"))

required_vars <- model_vars
if (!is.null(opt$strata)) {
  required_vars <- unique(c(required_vars, opt$strata))
}

existing_vars <- intersect(required_vars, colnames(metadata_filtered))

# -----------------------------
# automatic type coercion
# -----------------------------
metadata_filtered <- metadata_filtered %>%
  mutate(across(all_of(existing_vars), ~ {
    if (is.character(.)) as.factor(.) else .
  }))

# -----------------------------
# complete-case filtering
# -----------------------------
metadata_complete <- metadata_filtered %>%
  drop_na(all_of(existing_vars))

abundance_final <- abundance_filtered[rownames(metadata_complete), , drop = FALSE]

# -----------------------------
# insufficient data safeguard
# -----------------------------
write_empty_result <- function(path) {
  df <- data.frame(
    source = c("Residual", "Total"),
    Df = c(0, 0),
    SumOfSqs = c(0, 0),
    R2 = c(0, 0),
    F = c(NA, NA),
    Pr..F. = c(NA, NA)
  )
  write.table(df, file=path, sep="\t", quote=FALSE, row.names=FALSE)
}

if (nrow(metadata_complete) < 2 ||
    length(unique(metadata_complete[[opt$timepoint_col]])) < 2) {
  write_empty_result(opt$output)
  quit(save="no", status=0)
}

# -----------------------------
# distance matrix
# -----------------------------
dist_matrix <- vegdist(abundance_final, method = opt$distance)

if (length(unique(as.vector(dist_matrix))) == 1) {
  write_empty_result(opt$output)
  quit(save="no", status=0)
}

# -----------------------------
# run adonis2
# -----------------------------
formula_obj <- as.formula(paste("dist_matrix ~", opt$formula))

adonis_res <- tryCatch({
  if (!is.null(opt$strata)) {
    adonis2(
      formula = formula_obj,
      data = metadata_complete,
      permutations = 999,
      strata = metadata_complete[[opt$strata]],
      by = "margin"
    )
  } else {
    adonis2(
      formula = formula_obj,
      data = metadata_complete,
      permutations = 999,
      by = "margin"
    )
  }
}, error = function(e) NULL)

# -----------------------------
# failure fallback
# -----------------------------
if (is.null(adonis_res)) {
  write_empty_result(opt$output)
  quit(save="no", status=0)
}

# -----------------------------
# output
# -----------------------------
results_df <- as.data.frame(adonis_res) %>%
  rownames_to_column("source")

write.table(results_df,
            file = opt$output,
            sep = "\t",
            quote = FALSE,
            row.names = FALSE)

message("PERMANOVA complete. Results written to: ", opt$output)

