#!/usr/bin/env Rscript

# adonis.R — Clean version with no debugging header output
# Performs PERMANOVA (adonis2) for two timepoints with optional strata

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(vegan))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(tidyr))

option_list = list(
  make_option(c("--input"), type="character", default=NULL, help="Input abundance TSV file (sampleID as first column)."),
  make_option(c("--metadata"), type="character", default=NULL, help="Input metadata TSV file (sampleID as first column)."),
  make_option(c("--formula"), type="character", default=NULL, help="PERMANOVA formula string."),
  make_option(c("--distance"), type="character", default="bray", help="Distance metric."),
  make_option(c("--strata"), type="character", default=NULL, help="Optional column name for subject ID to use as strata."),
  make_option(c("--timepoint_a"), type="character", default=NULL, help="First timepoint."),
  make_option(c("--timepoint_b"), type="character", default=NULL, help="Second timepoint."),
  make_option(c("--output"), type="character", default="permanova_results.tsv", help="Output TSV file.")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# required
if (is.null(opt$input) || is.null(opt$metadata) || is.null(opt$formula) ||
    is.null(opt$timepoint_a) || is.null(opt$timepoint_b)) {
  stop("Input, metadata, formula, and both timepoints must be provided.", call.=FALSE)
}

# -----------------------------
# load data
# -----------------------------
abundance <- read_tsv(opt$input, show_col_types = FALSE)
metadata  <- read_tsv(opt$metadata, show_col_types = FALSE)

# Coerce metadata types
metadata <- metadata %>%
  mutate(
    timepoint = as.factor(timepoint),
    Intervention.Control_x = if ("Intervention.Control_x" %in% colnames(.)) as.factor(Intervention.Control_x) else Intervention.Control_x,
    Parous = if ("Parous" %in% colnames(.)) as.factor(Parous) else Parous,
    Maternal.age = if ("Maternal.age" %in% colnames(.)) as.numeric(Maternal.age) else Maternal.age,
    Maternal.BMI = if ("Maternal.BMI" %in% colnames(.)) as.numeric(Maternal.BMI) else Maternal.BMI
  )

# rownames
abundance_df <- abundance %>% column_to_rownames("sampleID")
metadata_df  <- metadata %>% column_to_rownames("sampleID")

# -----------------------------
# inner join
# -----------------------------
common_samples <- intersect(rownames(abundance_df), rownames(metadata_df))
abundance_df <- abundance_df[common_samples, , drop = FALSE]
metadata_df  <- metadata_df[common_samples, , drop = FALSE]

# -----------------------------
# subset by timepoints
# -----------------------------
metadata_filtered <- metadata_df %>% filter(timepoint %in% c(opt$timepoint_a, opt$timepoint_b))
abundance_filtered <- abundance_df[rownames(metadata_filtered), , drop = FALSE]

metadata_filtered$timepoint <- droplevels(metadata_filtered$timepoint)
if ("Intervention.Control_x" %in% colnames(metadata_filtered)) metadata_filtered$Intervention.Control_x <- droplevels(metadata_filtered$Intervention.Control_x)
if ("Parous" %in% colnames(metadata_filtered)) metadata_filtered$Parous <- droplevels(metadata_filtered$Parous)

# -----------------------------
# complete-case filter
# -----------------------------
formula_str <- gsub("\\s+", "", opt$formula)
model_vars <- unlist(strsplit(formula_str, "\\+"))

required_vars <- model_vars
if (!is.null(opt$strata)) required_vars <- unique(c(required_vars, opt$strata))

vars_exist <- intersect(required_vars, colnames(metadata_filtered))

metadata_complete <- metadata_filtered %>% drop_na(all_of(vars_exist))
abundance_final <- abundance_filtered[rownames(metadata_complete), , drop = FALSE]

# If insufficient samples, output empty table format
if (nrow(metadata_complete) < 2 || length(unique(metadata_complete$timepoint)) < 2) {
  results_df <- data.frame(
    source=c("Residual","Total"),
    Df=c(0,0), SumOfSqs=c(0,0), R2=c(0,0),
    F=c(NA,NA), Pr..F.=c(NA,NA)
  )
  write.table(results_df, file=opt$output, sep="\t", quote=FALSE, row.names=FALSE)
  quit(save="no", status=0)
}

# -----------------------------
# distance matrix
# -----------------------------
dist_matrix <- vegdist(abundance_final, method = opt$distance)

if (length(unique(as.vector(dist_matrix))) == 1) {
  results_df <- data.frame(
    source=c("Residual","Total"),
    Df=c(0,0), SumOfSqs=c(0,0), R2=c(0,0),
    F=c(NA,NA), Pr..F.=c(NA,NA)
  )
  write.table(results_df, file=opt$output, sep="\t", quote=FALSE, row.names=FALSE)
  quit(save="no", status=0)
}

# -----------------------------
# run adonis2
# -----------------------------
formula_obj <- as.formula(paste("dist_matrix ~", opt$formula))

adonis_res <- NULL
try({
  if (!is.null(opt$strata)) {
    adonis_res <- adonis2(
      formula=formula_obj,
      data=metadata_complete,
      permutations=999,
      method=opt$distance,
      strata=metadata_complete[[opt$strata]],
      by="margin"
    )
  } else {
    adonis_res <- adonis2(
      formula=formula_obj,
      data=metadata_complete,
      permutations=999,
      method=opt$distance,
      by="margin"
    )
  }
}, silent=TRUE)

# If adonis2 fails
if (is.null(adonis_res)) {
  results_df <- data.frame(
    source=c("Residual","Total"),
    Df=c(0,0), SumOfSqs=c(0,0), R2=c(0,0),
    F=c(NA,NA), Pr..F.=c(NA,NA)
  )
  write.table(results_df, file=opt$output, sep="\t", quote=FALSE, row.names=FALSE)
  quit(save="no", status=0)
}

# -----------------------------
# output results
# -----------------------------
results_df <- as.data.frame(adonis_res) %>% rownames_to_column("source")

if (file.exists(opt$output)) file.remove(opt$output)
write.table(results_df, file=opt$output, sep="\t", quote=FALSE, row.names=FALSE)

message(paste("PERMANOVA complete. Results written to:", opt$output))

