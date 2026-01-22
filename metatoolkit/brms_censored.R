#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(brms)
  library(dplyr)
  library(tidyr)
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
  make_option(c("--lod"), type = "double",
              help = "Limit of detection (numeric, on original scale)"),
  make_option(c("--baseline"), type = "character", default = NULL,
              help = "Baseline levels: var=level,var=level"),
  make_option(c("--order"), type = "character", default = NULL,
              help = "Factor order: var=level1,level2,...;var=level1,level2"),
  make_option(c("--iter"), type = "integer", default = 4000,
              help = "Total MCMC iterations (default: 4000)"),
  make_option(c("--chains"), type = "integer", default = 4,
              help = "Number of chains (default: 4)")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$lod)) {
  stop("--lod must be supplied for censored models")
}

# -------------------------
# Factor control helper
# -------------------------
apply_factor_controls <- function(df, baseline = NULL, order = NULL) {

  if (!is.null(order)) {
    var_specs <- strsplit(order, ";")[[1]]
    for (spec in var_specs) {
      parts <- strsplit(spec, "=")[[1]]
      var   <- parts[1]
      lvls  <- strsplit(parts[2], ",")[[1]]
      if (var %in% colnames(df)) {
        df[[var]] <- factor(trimws(df[[var]]), levels = lvls)
      }
    }
  }

  if (!is.null(baseline)) {
    specs <- strsplit(baseline, ",")[[1]]
    for (spec in specs) {
      parts <- strsplit(spec, "=")[[1]]
      var <- parts[1]
      ref <- parts[2]
      if (var %in% colnames(df)) {
        df[[var]] <- relevel(factor(df[[var]]), ref = ref)
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

# -------------------------
# Model loop
# -------------------------
for (resp in response_vars) {

  df <- data %>%
    select(any_of(c(colnames(df_meta), resp, opt$group))) %>%
    rename(response = all_of(resp)) %>%
    filter(!is.na(response))

  df <- apply_factor_controls(
    df,
    baseline = opt$baseline,
    order    = opt$order
  )

  # ---- log-transform inside model ----
  df <- df %>%
    mutate(
      log_response = log(response),
      cens = ifelse(response < opt$lod, "left", "none"),
      log_response = ifelse(response < opt$lod, log(opt$lod), log_response)
    )

  full_formula <- paste0(
    "log_response | cens(cens) ~ ",
    opt$formula,
    " + (1 | ", opt$group, ")"
  )
  
  model_formula <- bf(as.formula(full_formula))

  fit <- brm(
    model_formula,
    data = df,
    family = gaussian(),
    chains = opt$chains,
    iter = opt$iter,
    cores = opt$chains,
    refresh = 0,
    prior = c(
      prior(normal(0, 2), class = "b"),
      prior(student_t(3, 0, 2.5), class = "sd"),
      prior(student_t(3, 0, 2.5), class = "sigma")
    )
  )

  tidy_out <- as.data.frame(fixef(fit, probs = c(0.025, 0.975))) %>%
    tibble::rownames_to_column("term") %>%
    rename(
      estimate = Estimate,
      conf.low = Q2.5,
      conf.high = Q97.5
    ) %>%
    mutate(
      response_variable = resp,
      n_obs    = nrow(df),
      n_groups = dplyr::n_distinct(df[[opt$group]])
    ) %>%
    select(
      response_variable,
      term,
      estimate,
      conf.low,
      conf.high,
      n_obs,
      n_groups
    )

  results[[resp]] <- tidy_out
}

# -------------------------
# Write output
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

cat("Censored brms analysis complete.\n")
cat("Results written to:", opt$output, "\n")

