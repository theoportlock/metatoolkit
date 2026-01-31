#!/usr/bin/env Rscript

# Generic Beta Diversity Pipeline (PERMANOVA + PERMDISP + Pairwise)
# Supports:
#   - Explicit factor ordering & baselines (via --order, --baseline)
#   - Mixed-effects style strata (restricted permutations)
#   - Interaction terms in formulas
#   - Automatic sample alignment
#   - Tidy output

suppressPackageStartupMessages({
  library(optparse)
  library(vegan)
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(stringr)
})

# ---------------------------------------------------------
# CLI options
# ---------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--input"), type = "character",
              help = "Input Abundance TSV (Samples x Taxa or Taxa x Samples)"),
  make_option(c("-m", "--metadata"), type = "character",
              help = "Metadata TSV (first column = sample ID)"),
  make_option(c("-f", "--formula"), type = "character",
              help = "Formula (e.g. 'timepoint * group + sex')"),
  make_option(c("-s", "--strata"), type = "character", default = NULL,
              help = "Strata/Block variable for restricted permutations (e.g. subjectID)"),
  make_option(c("-d", "--distance"), type = "character", default = "bray",
              help = "Distance metric (bray, jaccard, euclidean, robust.aitchison)"),
  make_option(c("-o", "--output"), type = "character", default = "beta_stats.tsv",
              help = "Output TSV file"),
  make_option(c("--baseline"), type = "character", default = NULL,
              help = "Baseline levels: var=level,var=level"),
  make_option(c("--order"), type = "character", default = NULL,
              help = "Factor order: var=level1,level2,...;var=level1,level2")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Check required
if (is.null(opt$input) || is.null(opt$metadata) || is.null(opt$formula)) {
  stop("Missing required arguments. Usage: Rscript beta_generic.R -i abund.tsv -m meta.tsv -f 'formula'")
}

# ---------------------------------------------------------
# Factor control helper (From your GLMM script)
# ---------------------------------------------------------
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
        if (ref %in% levels(df[[var]])) {
          df[[var]] <- relevel(df[[var]], ref = ref)
        }
      }
    }
  }
  return(df)
}

# ---------------------------------------------------------
# Data Loading & Alignment
# ---------------------------------------------------------
message("Loading data...")
abundance <- read_tsv(opt$input, show_col_types = FALSE)
metadata  <- read_tsv(opt$metadata, show_col_types = FALSE)

# 1. Handle Sample IDs in Metadata
# Assume first column is ID
meta_id_col <- colnames(metadata)[1]
metadata <- metadata %>% column_to_rownames(meta_id_col)

# 2. Handle Abundance Orientation
# If first column is numeric, assume it's a matrix. If character, it's names.
if (is.character(abundance[[1]])) {
  # Save names
  row_names <- abundance[[1]]
  abundance <- abundance %>% select(-1) %>% as.data.frame()
  rownames(abundance) <- row_names
}

# Check orientation: We need Samples as ROWS.
# If abundance rownames match metadata rownames, we are good.
# If abundance colnames match metadata rownames, we need to transpose.
if (any(colnames(abundance) %in% rownames(metadata))) {
  message("Transposing abundance table to Sample x Feature format...")
  abundance <- t(abundance)
}

# 3. Align Samples (Intersect)
common <- intersect(rownames(abundance), rownames(metadata))
if (length(common) == 0) stop("No matching sample IDs found!")

abundance <- abundance[common, , drop=FALSE]
metadata  <- metadata[common, , drop=FALSE]
message(paste("Aligned samples:", length(common)))

# ---------------------------------------------------------
# Prepare Variables
# ---------------------------------------------------------
# Extract vars from formula string
raw_vars <- unique(unlist(strsplit(gsub("\\s+", "", opt$formula), "[+*:]")))
# Add strata if present
if(!is.null(opt$strata)) raw_vars <- c(raw_vars, opt$strata)

# Ensure vars exist
missing_vars <- setdiff(raw_vars, colnames(metadata))
if(length(missing_vars) > 0) stop(paste("Variables not found in metadata:", paste(missing_vars, collapse=", ")))

# Apply Factor Controls
metadata <- apply_factor_controls(metadata, baseline = opt$baseline, order = opt$order)

# Ensure formula vars are factors where appropriate (if character)
for(v in raw_vars) {
  if(is.character(metadata[[v]])) metadata[[v]] <- as.factor(metadata[[v]])
}

# Drop NAs in model variables
metadata <- metadata %>% drop_na(all_of(raw_vars))
abundance <- abundance[rownames(metadata), , drop=FALSE]

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------
results_list <- list()

# 1. Calculate Distance Matrix
message(paste("Calculating", opt$distance, "distance matrix..."))
dist_mat <- vegdist(abundance, method = opt$distance)

# ---------------------------------------------------------
# Test 1: Global PERMANOVA (Centroid)
# ---------------------------------------------------------
message("Running Global PERMANOVA...")
perm_formula <- as.formula(paste("dist_mat ~", opt$formula))
perm_res <- adonis2(
  perm_formula,
  data = metadata,
  permutations = 999,
  strata = if(!is.null(opt$strata)) metadata[[opt$strata]] else NULL,
  by = "margin" # Marginal effects (Type III SS equivalent)
)

perm_df <- as.data.frame(perm_res) %>%
  rownames_to_column("term") %>%
  filter(!is.na(F)) %>%
  transmute(
    test_type = "PERMANOVA",
    comparison = "Global",
    term = term,
    Df = Df,
    statistic = F,
    p_value = `Pr(>F)`,
    R2 = R2
  )

results_list[["permanova"]] <- perm_df

# ---------------------------------------------------------
# Test 2: Global PERMDISP (Dispersion)
# ---------------------------------------------------------
message("Running Global PERMDISP...")
# We run betadisper for each main effect and interaction term in the formula
# Note: betadisper takes a single group vector. For interactions, we combine columns.

disp_results <- list()
terms <- attr(terms(perm_formula), "term.labels")

for (t in terms) {
  # Clean term name
  clean_t <- t

  # Define grouping vector
  if (grepl(":", t)) {
    # Interaction term: create composite factor
    parts <- strsplit(t, ":")[[1]]
    group_vec <- interaction(metadata[, parts], drop=TRUE)
  } else {
    # Main effect
    group_vec <- metadata[[t]]
  }

  # Only run if factor
  if (is.factor(group_vec) || is.character(group_vec)) {
    tryCatch({
      mod <- betadisper(dist_mat, group_vec)
      # ANOVA on the distances
      ano <- anova(lm(mod$distances ~ group_vec))

      disp_results[[t]] <- data.frame(
        test_type = "PERMDISP",
        comparison = "Global",
        term = t,
        Df = ano$Df[1],
        statistic = ano$`F value`[1],
        p_value = ano$`Pr(>F)`[1],
        R2 = NA
      )
    }, error = function(e) {
      message(paste("Skipping PERMDISP for", t, "- possibly numeric or singular."))
    })
  }
}
results_list[["permdisp"]] <- bind_rows(disp_results)

# ---------------------------------------------------------
# Test 3: Pairwise Comparisons (If interaction or factor > 2 levels)
# ---------------------------------------------------------
message("Running Pairwise Comparisons...")

# We use a custom helper to run pairwise adonis
# We specifically look for the Reference level defined in options, if any.

# ---------------------------------------------------------
# Helper: Pairwise PERMANOVA (Silenced)
# ---------------------------------------------------------
run_pairwise_adonis <- function(dist_obj, meta_df, group_col, strata_col = NULL) {
  # Get all pairs
  lvls <- levels(as.factor(meta_df[[group_col]]))
  if (length(lvls) < 2) return(NULL)

  # Check if baseline reference exists in options
  ref_level <- NULL
  if (!is.null(opt$baseline)) {
    specs <- strsplit(opt$baseline, ",")[[1]]
    for (s in specs) {
      parts <- strsplit(s, "=")[[1]]
      if (parts[1] == group_col) ref_level <- parts[2]
    }
  }

  pairs <- combn(lvls, 2)
  res_list <- list()

  for(i in 1:ncol(pairs)) {
    p1 <- pairs[1, i]
    p2 <- pairs[2, i]

    # Filter: If baseline is set, skip pairs that don't include it
    if (!is.null(ref_level)) {
      if (p1 != ref_level && p2 != ref_level) next
    }

    # Subset Data
    idx <- meta_df[[group_col]] %in% c(p1, p2)
    sub_meta <- meta_df[idx, ]

    # Subset Distance Matrix (Crucial step for pairwise)
    # converting dist -> matrix -> subset -> dist
    sub_dist <- as.dist(as.matrix(dist_obj)[idx, idx])

    # Run Test (Silencing the 'complete enumeration' messages)
    try({
      # suppressMessages() hides the "Set of permutations < minperm" spam
      ad <- suppressMessages(
        adonis2(sub_dist ~ sub_meta[[group_col]],
                permutations = 999,
                strata = if(!is.null(strata_col)) sub_meta[[strata_col]] else NULL)
      )

      res_list[[length(res_list)+1]] <- data.frame(
        test_type = "Pairwise_PERMANOVA",
        comparison = paste(p1, "vs", p2),
        term = group_col,
        Df = ad$Df[1],
        statistic = ad$F[1],
        p_value = ad$`Pr(>F)`[1],
        R2 = ad$R2[1]
      )
    }, silent=TRUE)
  }
  bind_rows(res_list)
}

pairwise_out <- list()

# 1. Main effects pairwise
vars_to_test <- raw_vars[raw_vars %in% colnames(metadata)]
for (v in vars_to_test) {
  if (is.factor(metadata[[v]]) && nlevels(metadata[[v]]) > 2) {
    pairwise_out[[v]] <- run_pairwise_adonis(dist_mat, metadata, v, opt$strata)
  }
}

# 2. Interaction slicing (e.g. Timepoint WITHIN Group)
# If formula has interaction "A * B", we test A within levels of B, and B within levels of A
if (any(grepl(":", terms))) {
  parts <- strsplit(grep(":", terms, value=TRUE)[1], ":")[[1]] # Take first interaction
  v1 <- parts[1]; v2 <- parts[2]

  # Test v1 within v2
  if (v2 %in% colnames(metadata)) {
    for (lvl in unique(metadata[[v2]])) {
      sub_idx <- metadata[[v2]] == lvl
      sub_meta <- metadata[sub_idx, ]
      sub_dist <- as.dist(as.matrix(dist_mat)[sub_idx, sub_idx])

      # Run pairwise on v1 for this subset
      pw <- run_pairwise_adonis(sub_dist, sub_meta, v1, opt$strata)
      if(!is.null(pw)) {
        pw$term <- paste0(v1, " (in ", v2, "=", lvl, ")")
        pairwise_out[[paste(v1, v2, lvl)]] <- pw
      }
    }
  }
}

results_list[["pairwise"]] <- bind_rows(pairwise_out)

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
final_df <- bind_rows(results_list) %>%
  select(test_type, term, comparison, Df, statistic, R2, p_value) %>%
  mutate(across(c(statistic, R2, p_value), \(x) round(x, 4)))

# Create directory
out_dir <- dirname(opt$output)
if (!dir.exists(out_dir) && out_dir != ".") dir.create(out_dir, recursive = TRUE)

write_tsv(final_df, opt$output)

message("------------------------------------------------")
message("Analysis Complete.")
message(paste("Global PERMANOVA R2:", round(perm_df$R2[1], 3)))
message(paste("Results saved to:", opt$output))
message("------------------------------------------------")
