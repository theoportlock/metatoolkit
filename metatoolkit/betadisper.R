#!/usr/bin/env Rscript

# ------------------------------------------------------------
# BETADISPER (Dispersion of Beta Diversity)
# MaAsLin2-style outputs (term-level)
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
  make_option(c("--outdir"), type="character", default="betadisper_output"),
  make_option(c("--reference"), action="append", type="character",
              help="Set reference: 'Variable,Level'. Repeatable."),
  make_option(c("--seed"), type="integer", default=123),
  make_option(c("--permutations"), type="integer", default=999),
  make_option(c("--pairwise"), action="store_true", default=FALSE,
              help="Run Tukey HSD pairwise comparisons")
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

model_terms <- attr(terms(as.formula(paste("~", opt$formula))), "term.labels")

metadata_complete <- metadata_df %>%
  drop_na(all_of(all.vars(as.formula(paste("~", opt$formula))))) %>%
  mutate(across(everything(),
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
# Helper: build grouping variable
# ------------------------------------------------------------

build_group <- function(term, metadata) {

  if (str_detect(term, ":")) {
    vars <- str_split(term, ":", simplify=TRUE)
    group <- interaction(metadata[, vars], drop=TRUE)
  } else {
    group <- metadata[[term]]
  }

  return(as.factor(group))
}

# ------------------------------------------------------------
# Run betadisper per term
# ------------------------------------------------------------

results <- list()
pairwise_results <- list()

for (term in model_terms) {

  message("Running betadisper for: ", term)

  group <- build_group(term, metadata_complete)

  if (length(unique(group)) < 2) {
    message("Skipping ", term, " (only one group)")
    next
  }

  bd <- betadisper(dist_final, group)

  # ANOVA test
  an <- anova(bd)

  # Permutation test (more robust)
  perm <- permutest(bd, permutations=opt$permutations)

  res_row <- tibble(
    Term = term,
    Df = an$Df[1],
    SumSq = an$`Sum Sq`[1],
    MeanSq = an$`Mean Sq`[1],
    F = an$`F value`[1],
    P_value = perm$tab[1, "Pr(>F)"]
  )

  results[[term]] <- res_row

  # Optional pairwise
  if (opt$pairwise) {
    tuk <- TukeyHSD(bd)
    tuk_df <- as.data.frame(tuk$group) %>%
      rownames_to_column("Comparison") %>%
      mutate(Term = term)

    pairwise_results[[term]] <- tuk_df
  }
}

# ------------------------------------------------------------
# Save Outputs
# ------------------------------------------------------------

final_res <- bind_rows(results)

write_tsv(final_res,
          file.path(opt$outdir, "betadisper_results.tsv"))

if (opt$pairwise && length(pairwise_results) > 0) {
  pw <- bind_rows(pairwise_results)
  write_tsv(pw,
            file.path(opt$outdir, "betadisper_pairwise.tsv"))
}

write_lines(final_samples,
            file.path(opt$outdir, "samples_included.txt"))

message("Done.")
message("Results written to: ", opt$outdir)
