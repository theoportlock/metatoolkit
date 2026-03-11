#!/usr/bin/env Rscript

# ------------------------------------------------------------
# PERMANOVA (adonis2) – Model Matrix / Coefficient-Level Version
# MaAsLin2-style outputs
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(vegan)
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(stringr)
})

# ------------------------------------------------------------
# CLI Options
# ------------------------------------------------------------

option_list <- list(
  make_option(c("--distances"), type="character", help="Long-format distance TSV."),
  make_option(c("--metadata"), type="character", help="Metadata TSV."),
  make_option(c("--formula"), type="character",
              help="Model formula RHS (e.g. 'Intervention * timepoint + Age')."),
  make_option(c("--source_col"), type="character", default="source"),
  make_option(c("--target_col"), type="character", default="target"),
  make_option(c("--dist_col"), type="character", default="distance"),
  make_option(c("--sample_id_col"), type="character", default="sampleID"),
  make_option(c("--outdir"), type="character", default="permanova_output"),
  make_option(c("--strata"), type="character", default=NULL),
  make_option(c("--reference"), action="append", type="character",
              help="Set reference: 'Variable,Level'. Repeatable."),
  make_option(c("--seed"), type="integer", default=123),
  make_option(c("--permutations"), type="integer", default=999)
)

opt <- parse_args(OptionParser(option_list=option_list))

required <- c("distances","metadata","formula")
missing <- required[sapply(required, function(x) is.null(opt[[x]]))]
if (length(missing) > 0)
  stop(paste("Missing required:", paste(missing, collapse=", ")), call.=FALSE)

if (!dir.exists(opt$outdir))
  dir.create(opt$outdir, recursive=TRUE)

set.seed(opt$seed)

# ------------------------------------------------------------
# Load Distance Matrix
# ------------------------------------------------------------

dist_long <- read_tsv(opt$distances, show_col_types=FALSE)

dist_wide <- dist_long %>%
  select(all_of(c(opt$source_col,opt$target_col,opt$dist_col))) %>%
  pivot_wider(names_from = !!sym(opt$target_col),
              values_from = !!sym(opt$dist_col)) %>%
  column_to_rownames(opt$source_col) %>%
  as.matrix()

dist_wide[upper.tri(dist_wide)] <- t(dist_wide)[upper.tri(dist_wide)]
dist_wide[is.na(dist_wide)] <- 0

# ------------------------------------------------------------
# Load Metadata
# ------------------------------------------------------------

metadata_df <- read_tsv(opt$metadata, show_col_types=FALSE) %>%
  column_to_rownames(opt$sample_id_col)

common_samples <- intersect(rownames(dist_wide), rownames(metadata_df))
dist_wide <- dist_wide[common_samples, common_samples]
metadata_df <- metadata_df[common_samples, , drop=FALSE]

if (nrow(metadata_df) < 2)
  stop("Insufficient overlapping samples.", call.=FALSE)

# ------------------------------------------------------------
# Clean Metadata + Apply References
# ------------------------------------------------------------

model_vars <- all.vars(as.formula(paste("~", opt$formula)))
if (!is.null(opt$strata))
  model_vars <- unique(c(model_vars, opt$strata))

metadata_complete <- metadata_df %>%
  drop_na(all_of(model_vars)) %>%
  mutate(across(all_of(model_vars),
         ~ if(is.character(.)) as.factor(.) else .))

if (!is.null(opt$reference)) {
  for (ref in opt$reference) {
    parts <- str_split(ref, ",", simplify=TRUE)
    var <- str_trim(parts[1])
    lev <- str_trim(parts[2])
    if (var %in% colnames(metadata_complete)) {
      metadata_complete[[var]] <-
        relevel(as.factor(metadata_complete[[var]]), ref=lev)
    }
  }
}

final_samples <- rownames(metadata_complete)
dist_final <- as.dist(dist_wide[final_samples, final_samples])

# ------------------------------------------------------------
# Construct Explicit Model Matrix
# ------------------------------------------------------------

design_formula <- as.formula(paste("~", opt$formula))
X <- model.matrix(design_formula, data=metadata_complete)

# Remove intercept (critical)
if ("(Intercept)" %in% colnames(X))
  X <- X[, colnames(X) != "(Intercept)", drop=FALSE]

X_df <- as.data.frame(X)

# ------------------------------------------------------------
# Run Coefficient-Level PERMANOVA
# ------------------------------------------------------------

run_adonis <- function(by_type) {

  message("Running coefficient-level adonis2 (by = ", by_type, ")")

  args <- list(
    formula = as.formula("dist_final ~ ."),
    data = X_df,
    permutations = opt$permutations,
    by = by_type
  )

  if (!is.null(opt$strata))
    args$strata <- metadata_complete[[opt$strata]]

  tryCatch({
    do.call(adonis2, args)
  }, error=function(e){
    message("adonis2 error: ", e$message)
    return(NULL)
  })
}

res_sequential <- run_adonis("terms")
res_marginal   <- run_adonis("margin")

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------

save_res <- function(res, name) {
  if (is.null(res)) return()
  out <- as.data.frame(res) %>%
    rownames_to_column("Coefficient")
  write_tsv(out, file.path(opt$outdir, paste0(name,".tsv")))
}

save_res(res_sequential, "permanova_sequential")
save_res(res_marginal,   "permanova_marginal")

write_lines(final_samples,
            file.path(opt$outdir, "samples_included.txt"))

message("Done.")
message("Results written to: ", opt$outdir)
