#!/usr/bin/env Rscript

# GLMM using lmerTest / lme4
# Supports:
#   - multiple response variables
#   - explicit factor ordering
#   - explicit baseline (reference) levels
#   - gaussian / poisson / negbin families
#   - optional saving of fitted model objects (for posthoc contrasts)

suppressPackageStartupMessages({
  library(optparse)
  library(lmerTest)
  library(lme4)
  library(broom.mixed)
  library(dplyr)
})

# -------------------------
# CLI options
# -------------------------
option_list <- list(
  make_option(c("-i", "--input"), type = "character",
              help = "Input TSV file (first column = sample ID)"),
  make_option(c("-m", "--metadata"), type = "character",
              help = "Metadata TSV file (first column = sample ID)"),
  make_option(c("-f", "--formula"), type = "character",
              help = "Fixed effects formula (e.g. 'timepoint * group + sex')"),
  make_option(c("-g", "--group"), type = "character",
              help = "Random effect grouping variable (e.g. subjectID)"),
  make_option(c("-o", "--output"), type = "character",
              help = "Output TSV file"),
  make_option(c("--family"), type = "character", default = "gaussian",
              help = "Family: gaussian | poisson | negbin (default: gaussian)"),
  make_option(c("--zscore"), action = "store_true", default = FALSE,
              help = "Z-score scale the response variable"),
  make_option(c("--baseline"), type = "character", default = NULL,
              help = "Baseline levels: var=level,var=level"),
  make_option(c("--order"), type = "character", default = NULL,
              help = "Factor order: var=level1,level2,...;var=level1,level2"),
  make_option(c("--save-models"), dest = "save_models",
	      type = "character", default = NULL,
	      help = "Optional RDS file to save fitted model objects")
)

opt <- parse_args(OptionParser(option_list = option_list))

# -------------------------
# Factor control helper
# -------------------------
apply_factor_controls <- function(df, baseline = NULL, order = NULL) {

  # ---- Apply explicit ordering ----
  if (!is.null(order)) {
    var_specs <- strsplit(order, ";")[[1]]

    for (spec in var_specs) {
      parts <- strsplit(spec, "=")[[1]]
      var   <- parts[1]
      lvls  <- strsplit(parts[2], ",")[[1]]

      if (var %in% colnames(df)) {

        df[[var]] <- trimws(as.character(df[[var]]))

        missing <- setdiff(unique(df[[var]]), lvls)
        if (length(missing) > 0) {
          stop(
            paste0(
              "Variable '", var,
              "' contains values not listed in --order: ",
              paste(missing, collapse = ", ")
            )
          )
        }

        df[[var]] <- factor(df[[var]], levels = lvls)
      }
    }
  }

  # ---- Apply baseline (reference) levels ----
  if (!is.null(baseline)) {
    specs <- strsplit(baseline, ",")[[1]]

    for (spec in specs) {
      parts <- strsplit(spec, "=")[[1]]
      var   <- parts[1]
      ref   <- parts[2]

      if (var %in% colnames(df)) {

        if (!is.factor(df[[var]])) {
          df[[var]] <- factor(trimws(as.character(df[[var]])))
        }

        if (!ref %in% levels(df[[var]])) {
          stop(
            paste0(
              "Baseline level '", ref,
              "' not found in variable '", var,
              "'. Levels are: ",
              paste(levels(df[[var]]), collapse = ", ")
            )
          )
        }

        df[[var]] <- relevel(df[[var]], ref = ref)
      }
    }
  }

  df
}

# -------------------------
# Load data
# -------------------------
df_data <- read.delim(opt$input, stringsAsFactors = FALSE, check.names = FALSE)
df_meta <- read.delim(opt$metadata, stringsAsFactors = FALSE, check.names = FALSE)

data_id_col <- colnames(df_data)[1]
meta_id_col <- colnames(df_meta)[1]

data <- merge(
  df_meta,
  df_data,
  by.x = meta_id_col,
  by.y = data_id_col
)

response_vars <- setdiff(colnames(df_data), data_id_col)
results <- list()
models  <- list()

# -------------------------
# Model loop
# -------------------------
for (resp in response_vars) {

  df <- data %>%
    select(any_of(c(colnames(df_meta), resp, opt$group))) %>%
    rename(response = all_of(resp)) %>%
    filter(!is.na(response))

  if (!(opt$group %in% colnames(df))) {
    stop("Grouping variable not found: ", opt$group)
  }

  # Apply factor controls
  df <- apply_factor_controls(
    df,
    baseline = opt$baseline,
    order    = opt$order
  )

  if (opt$zscore) {
    df$response <- as.numeric(scale(df$response))
  }

  model_formula <- as.formula(
    paste0("response ~ ", opt$formula, " + (1|", opt$group, ")")
  )

  model <- switch(
    opt$family,
    gaussian = lmer(model_formula, data = df, REML = FALSE),
    poisson = glmer(model_formula, data = df, family = poisson(link = "log")),
    negbin  = glmer.nb(model_formula, data = df),
    stop("Unsupported family: ", opt$family)
  )

  models[[resp]] <- model

  n_obs    <- nrow(df)
  n_groups <- dplyr::n_distinct(df[[opt$group]], na.rm = TRUE)

  tidy_out <- broom.mixed::tidy(
    model,
    effects  = "fixed",
    conf.int = TRUE
  ) %>%
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

# -------------------------
# Write outputs
# -------------------------
final <- bind_rows(results)

out_dir <- dirname(opt$output)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

write.table(
  final,
  opt$output,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

if (!is.null(opt$save_models)) {
  saveRDS(models, opt$save_models)
  cat("Model objects saved to:", opt$save_models, "\n")
}

cat("GLMM analysis complete.\n")
cat("Results written to:", opt$output, "\n")

