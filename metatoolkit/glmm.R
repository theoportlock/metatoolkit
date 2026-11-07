#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(lmerTest)
  library(lme4)
  library(broom.mixed)
  library(dplyr)
  library(MASS)
})

# -------------------------
# CLI options
# -------------------------
option_list <- list(
  make_option(c("-i", "--input"), type = "character"),
  make_option(c("-m", "--metadata"), type = "character"),
  make_option(c("-f", "--formula"), type = "character", default = NULL),
  make_option(c("-g", "--group"), type = "character", default = NULL),
  make_option(c("-o", "--output"), type = "character"),
  make_option(c("--family"), type = "character", default = "gaussian"),
  make_option(c("--zscore"), action = "store_true", default = FALSE),
  make_option(c("--baseline"), type = "character", default = NULL),
  make_option(c("--order"), type = "character", default = NULL),
  make_option(c("--save-models"), type = "character", default = NULL)
)

opt <- parse_args(OptionParser(option_list = option_list))

# -------------------------
# Helpers
# -------------------------
build_default_formula <- function(df_meta) {
  paste(colnames(df_meta), collapse = " + ")
}

apply_factor_controls <- function(df, baseline = NULL, order = NULL) {

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
          stop(paste0("Variable '", var, "' contains values not listed in --order: ",
                      paste(missing, collapse = ", ")))
        }

        df[[var]] <- factor(df[[var]], levels = lvls)
      }
    }
  }

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
          stop(paste0("Baseline level '", ref, "' not found in variable '", var, "'"))
        }

        df[[var]] <- stats::relevel(df[[var]], ref = ref)
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

data <- merge(df_meta, df_data,
              by.x = meta_id_col,
              by.y = data_id_col)

# -------------------------
# Formula handling
# -------------------------
if (is.null(opt$formula)) {
  opt$formula <- build_default_formula(df_meta[, -1, drop = FALSE])
  cat("No formula supplied → using all metadata columns:\n  ", opt$formula, "\n")
}

# -------------------------
# Group handling
# -------------------------
has_group <- !is.null(opt$group) && opt$group %in% colnames(data)

if (!is.null(opt$group) && !has_group) {
  stop("Grouping variable not found: ", opt$group)
}

if (has_group) {
  cat("Using Random Effect grouping variable:", opt$group, "\n")
} else {
  cat("No grouping variable supplied → using fixed-effects model\n")
}

# -------------------------
# Model loop
# -------------------------
response_vars <- setdiff(colnames(df_data), data_id_col)
results <- list()
models  <- list()

for (resp in response_vars) {

  cols_to_keep <- c(colnames(df_meta), resp)
  if (has_group) cols_to_keep <- c(cols_to_keep, opt$group)

  df <- data %>%
    dplyr::select(dplyr::any_of(cols_to_keep))

  if (!(resp %in% colnames(df))) {
    results[[resp]] <- data.frame(
      response_variable = resp,
      term = NA, estimate = NA, std.error = NA,
      statistic = NA, p.value = NA,
      conf.low = NA, conf.high = NA,
      n_obs = NA, n_groups = NA,
      status = "ERROR: response missing after merge/select"
    )
    next
  }

  df <- df %>%
    dplyr::rename(response = dplyr::all_of(resp))

  # Factor controls
  df <- tryCatch({
    apply_factor_controls(df, opt$baseline, opt$order)
  }, error = function(e) {
    results[[resp]] <<- data.frame(
      response_variable = resp,
      term = NA, estimate = NA, std.error = NA,
      statistic = NA, p.value = NA,
      conf.low = NA, conf.high = NA,
      n_obs = NA, n_groups = NA,
      status = paste("ERROR:", e$message)
    )
    return(NULL)
  })
  if (is.null(df)) next

  # NA filtering
  vars_in_formula <- all.vars(stats::as.formula(paste0("~", opt$formula)))
  cols_to_check <- c("response", vars_in_formula)
  if (has_group) cols_to_check <- c(cols_to_check, opt$group)

  df <- df %>%
    dplyr::filter(dplyr::if_all(dplyr::any_of(cols_to_check), ~ !is.na(.)))

  if (nrow(df) == 0) {
    results[[resp]] <- data.frame(
      response_variable = resp,
      term = NA, estimate = NA, std.error = NA,
      statistic = NA, p.value = NA,
      conf.low = NA, conf.high = NA,
      n_obs = NA, n_groups = NA,
      status = "ERROR: No data after NA filtering"
    )
    next
  }

  if (opt$zscore) {
    df$response <- as.numeric(scale(df$response))
  }

  # Build formula
  if (has_group) {
    model_formula <- stats::as.formula(
      paste0("response ~ ", opt$formula, " + (1|", opt$group, ")")
    )
  } else {
    model_formula <- stats::as.formula(
      paste0("response ~ ", opt$formula)
    )
  }

  # Fit model safely
  model <- tryCatch({

    if (opt$family == "gaussian") {
      if (has_group) {
        lmer(model_formula, data = df, REML = FALSE)
      } else {
        stats::lm(model_formula, data = df)
      }

    } else if (opt$family == "poisson") {
      if (has_group) {
        glmer(model_formula, data = df, family = stats::poisson(link = "log"))
      } else {
        stats::glm(model_formula, data = df, family = stats::poisson(link = "log"))
      }

    } else if (opt$family == "negbin") {
      if (has_group) {
        glmer.nb(model_formula, data = df)
      } else {
        MASS::glm.nb(model_formula, data = df)
      }

    } else {
      stop("Unsupported family: ", opt$family)
    }

  }, error = function(e) {
    results[[resp]] <<- data.frame(
      response_variable = resp,
      term = NA, estimate = NA, std.error = NA,
      statistic = NA, p.value = NA,
      conf.low = NA, conf.high = NA,
      n_obs = NA, n_groups = NA,
      status = paste("ERROR:", e$message)
    )
    return(NULL)
  })

  if (is.null(model)) next

  models[[resp]] <- model

  model_type <- ifelse(has_group, "GLMM", "LM/GLM")

  tidy_out <- broom.mixed::tidy(
    model,
    effects  = "fixed",
    conf.int = TRUE
  ) %>%
    dplyr::mutate(
      response_variable = resp,
      n_obs    = nrow(df),
      n_groups = ifelse(has_group,
                        dplyr::n_distinct(df[[opt$group]], na.rm = TRUE),
                        NA),
      status = paste("Converged (", model_type, ")", sep = "")
    ) %>%
    dplyr::select(
      response_variable,
      term,
      estimate,
      std.error,
      statistic,
      p.value,
      conf.low,
      conf.high,
      n_obs,
      n_groups,
      status
    )

  results[[resp]] <- tidy_out
}

# -------------------------
# Write output
# -------------------------
final <- dplyr::bind_rows(results)

out_dir <- dirname(opt$output)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

write.table(final, opt$output, sep = "\t", row.names = FALSE, quote = FALSE)

if (!is.null(opt$save_models)) {
  saveRDS(models, opt$save_models)
  cat("Model objects saved to:", opt$save_models, "\n")
}

cat("GLMM analysis complete.\n")
cat("Results written to:", opt$output, "\n")
