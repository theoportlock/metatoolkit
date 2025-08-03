#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(vegan)
  library(dplyr)
  library(readr)
})

option_list <- list(
  make_option(c("-y", "--response"), type = "character", help = "TSV file of response variables (e.g., species)"),
  make_option(c("-x", "--explanatory"), type = "character", help = "TSV file of explanatory variables (metadata)"),
  make_option(c("-f", "--fixed_effects"), type = "character", help = "Comma-separated list of fixed effect variable names (e.g., 'Age,Sex,Delivery'). If omitted, all variables in the explanatory file will be used.", default = NULL),
  make_option(c("-o", "--output"), type = "character", help = "Output TSV file")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Load data
Y <- read.table(opt$response, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
X <- read.table(opt$explanatory, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

# Match samples
common_samples <- intersect(rownames(Y), rownames(X))
Y <- Y[common_samples, , drop=FALSE]
X <- X[common_samples, , drop=FALSE]

# Handle missing data
X_cols_to_check <- if(is.null(opt$fixed_effects)) colnames(X) else trimws(strsplit(opt$fixed_effects, ",")[[1]])
if (length(X_cols_to_check) > 0) {
    complete_rows <- complete.cases(Y, X[, X_cols_to_check, drop = FALSE])
    if (sum(!complete_rows) > 0) {
      warning(paste0("Removing ", sum(!complete_rows), " samples with missing data."))
    }
    Y <- Y[complete_rows, , drop = FALSE]
    X <- X[complete_rows, , drop = FALSE]
}

# Parse fixed effects
if (is.null(opt$fixed_effects)) {
  fixed_effects <- colnames(X)
} else {
  fixed_effects <- trimws(strsplit(opt$fixed_effects, ",")[[1]])
}

# Check fixed_effects in X
missing_vars <- setdiff(fixed_effects, colnames(X))
if (length(missing_vars) > 0) {
  stop("The following fixed effects are missing from explanatory variables: ", paste(missing_vars, collapse = ", "))
}

# Remove variables with < 2 levels (constants)
kept_vars <- sapply(fixed_effects, function(v) length(unique(X[[v]])) > 1)
if (!all(kept_vars)) {
  removed_vars <- fixed_effects[!kept_vars]
  warning("Removing fixed effects with < 2 levels: ", paste(removed_vars, collapse = ", "))
}
fixed_effects <- fixed_effects[kept_vars]

if (length(fixed_effects) < 1) {
  stop("No fixed effects with >= 2 levels remain for analysis.")
}

# Subset X to fixed_effects only
X_sub <- X[, fixed_effects, drop = FALSE]

# --- MODIFICATION STARTS HERE ---
# Explicitly convert character columns to factors
# This prevents the `scale` function from being called on non-numeric data
for (col in colnames(X_sub)) {
  if (is.character(X_sub[[col]])) {
    X_sub[[col]] <- as.factor(X_sub[[col]])
    message(paste("Converted character column", col, "to factor."))
  }
}
# --- MODIFICATION ENDS HERE ---

# Build the full formula and run the full RDA model
full_formula <- as.formula(paste("Y ~", paste(paste0("`", fixed_effects, "`"), collapse = " + ")))
full_rda <- rda(full_formula, data = X_sub)

# Get marginal R2 and p-values for each term using anova.cca
anova_marginal <- anova.cca(full_rda, by = "margin", permutations = 999)

# Calculate the total variance for manual R2 calculation.
total_variance <- sum(anova_marginal$Variance, anova_marginal["Residual", "Variance"])

# Prepare results from the anova_marginal table
term_stats <- data.frame(
  explainer = rownames(anova_marginal),
  R2 = round(anova_marginal$Variance / total_variance, 5),
  adj_R2 = NA,
  pval = round(anova_marginal$`Pr(>F)`, 5),
  stringsAsFactors = FALSE
)

# Exclude the "Residual" row from the term statistics to avoid redundancy
term_stats <- term_stats[term_stats$explainer != "Residual", ]

# Get the total R2 and adjusted R2
total_constrained_r2 <- RsquareAdj(full_rda)$r.squared
full_adj_r2 <- RsquareAdj(full_rda)$adj.r.squared

# Add the final summary stats
summary_stats <- data.frame(
  explainer = c("Total_Constrained_R2", "Unconstrained_R2", "Adjusted_Full_Model"),
  R2 = c(round(total_constrained_r2, 5), round(1 - total_constrained_r2, 5), NA),
  adj_R2 = c(NA, NA, round(full_adj_r2, 5)),
  pval = c(NA, NA, NA),
  stringsAsFactors = FALSE
)

# Combine and reorder columns
final_df <- bind_rows(term_stats, summary_stats) %>%
  select(explainer, R2, adj_R2, pval)

# Write output TSV
write.table(final_df, file = opt$output, sep = "\t", row.names = FALSE, quote = FALSE)
