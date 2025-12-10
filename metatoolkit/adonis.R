#!/usr/bin/env Rscript

# PERMANOVA (adonis2) wrapper using vegan
suppressPackageStartupMessages({
  library(optparse)
  library(vegan)
  library(dplyr)
})


option_list <- list(
  make_option(c("-i", "--input"), type = "character",
              help = "Feature/abundance TSV file (samples x features)"),
  make_option(c("-m", "--metadata"), type = "character",
              help = 'Metadata TSV file'),
  make_option(c("-f", "--formula"), type = "character",
              help = 'Model formula, e.g. "Intervention * timepoint + Age"'),
  make_option(c("-d", "--distance"), type = "character", default = "bray",
              help = "Distance metric for vegdist (bray, jaccard, euclidean, etc). [default: bray]"),
  make_option(c("-o", "--output"), type = "character",
              help = "Output TSV file for PERMANOVA results")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Validate inputs
if (is.null(opt$input) || is.null(opt$metadata) || is.null(opt$formula) || is.null(opt$output)) {
  stop("Missing required arguments. Use --help for details.")
}

# Load data
feat <- read.delim(opt$input, stringsAsFactors = FALSE, check.names = FALSE)
meta <- read.delim(opt$metadata, stringsAsFactors = FALSE, check.names = FALSE)

# Merge by first column (sample ID)
feat_id <- colnames(feat)[1]
meta_id <- colnames(meta)[1]

data <- merge(meta, feat, by.x = meta_id, by.y = feat_id)

# Extract sample × feature matrix
feature_cols <- setdiff(colnames(feat), feat_id)
X <- data[, feature_cols, drop = FALSE]

# Compute distance matrix
dist_matrix <- vegdist(X, method = opt$distance)

# Construct PERMANOVA formula
adonis_formula <- as.formula(paste("dist_matrix ~", opt$formula))

# Run PERMANOVA (adonis2) with marginal significance (Type III)
perm <- adonis2(adonis_formula, data = data, permutations = 999, by = "margin")

# Convert results to data.frame for TSV output
perm_df <- as.data.frame(perm)
perm_df$term <- rownames(perm_df)
perm_df <- perm_df %>% select(term, everything())

# Write results
write.table(perm_df, opt$output, sep = "\t", row.names = FALSE, quote = FALSE)

cat("PERMANOVA (adonis2) analysis complete. Results written to:", opt$output, "\n")

