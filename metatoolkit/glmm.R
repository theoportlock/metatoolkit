#!/usr/bin/env Rscript

# GLMM for alpha diversity using lmerTest
# Mirrors mixedlm.py CLI structure but supports generalized linear mixed models
# Suitable for Richness / Faith's PD / Shannon with appropriate families
#
# Dependencies:
# install.packages(c("optparse", "lmerTest", "lme4", "broom.mixed", "dplyr"))

suppressPackageStartupMessages({
  library(optparse)
  library(lmerTest)
  library(lme4)
  library(broom.mixed)
  library(dplyr)
})

option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "Alpha diversity TSV file"),
  make_option(c("-m", "--metadata"), type = "character", help = "Metadata TSV file"),
  make_option(c("-f", "--formula"), type = "character", help = "Fixed effects formula e.g. 'Timepoint * Treatment + Age'"),
  make_option(c("-g", "--group"), type = "character", help = "Random effect grouping variable (e.g. SubjectID)"),
  make_option(c("-o", "--output"), type = "character", help = "Output TSV"),
  make_option(c("--family"), type = "character", default = "gaussian",
              help = "Family: gaussian | poisson | negbin (default: gaussian)"),
  make_option(c("--zscore"), action = "store_true", default = FALSE,
              help = "Z-score scale the response variable(s)")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Load data
alpha <- read.delim(opt$input, stringsAsFactors = FALSE, check.names = FALSE)
meta  <- read.delim(opt$metadata, stringsAsFactors = FALSE, check.names = FALSE)

# Merge by the first column of each table
alpha_id_col <- colnames(alpha)[1]
meta_id_col  <- colnames(meta)[1]

data <- merge(meta, alpha, by.x = meta_id_col, by.y = alpha_id_col)

# Identify response variables (all columns in alpha except the first column)
response_vars <- setdiff(colnames(alpha), alpha_id_col)

results <- list()

for (resp in response_vars) {

  df <- data %>%
    dplyr::select(dplyr::any_of(c(colnames(meta), resp, opt$group))) %>%
    dplyr::rename(response = all_of(resp)) %>%
    dplyr::filter(!is.na(response))

  if (!(opt$group %in% colnames(df))) {
    stop(paste("Grouping column", opt$group, "not found in merged data"))
  }

  if (opt$zscore) {
    df$response <- scale(df$response)
  }

  # Build model formula
  full_formula <- as.formula(paste0("response ~ ", opt$formula, " + (1|", opt$group, ")"))

  model <- NULL

  if (opt$family == "gaussian") {
    model <- lmer(full_formula, data = df, REML = FALSE)

  } else if (opt$family == "poisson") {
    model <- glmer(full_formula, data = df, family = poisson(link = "log"))

  } else if (opt$family == "negbin") {
    model <- glmer.nb(full_formula, data = df)

  } else {
    stop("Unsupported family: must be gaussian, poisson, or negbin")
  }

  # Precompute to avoid NSE environment issues
  n_obs    <- nrow(df)
  n_groups <- length(unique(df[[opt$group]]))
  
  tidy_out <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    mutate(
      response_variable = resp,
      n_obs    = n_obs,
      n_groups = n_groups
    ) %>%
    select(
      response_variable,
      term,
      estimate,
      std.error,
      statistic,
      p.value,
      conf.low,
      conf.high,
      n_obs,
      n_groups
    )

  results[[resp]] <- tidy_out
}


final <- bind_rows(results)
write.table(final, opt$output, sep = "\t", row.names = FALSE, quote = FALSE)

cat("GLMM analysis complete. Results written to:", opt$output, "\n")

