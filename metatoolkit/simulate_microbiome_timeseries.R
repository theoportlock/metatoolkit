#!/usr/bin/env Rscript
# Simulate a two-arm (control vs treatment) microbiome time series with SparseDOSSA2.
#
# Writes to out_dir:
#   counts.tsv        features x samples, simulated read counts
#   relabundance.tsv  features x samples, relative abundance
#   metadata.tsv      one row per sample (subject, arm, week, covariates)
#   spiked_truth.tsv  ground truth: spiked features, property, effect size
#
# Run with:  Rscript simulate_microbiome_timeseries.R

suppressPackageStartupMessages(library(SparseDOSSA2))

## ---- parameters ------------------------------------------------------------
seed              <- 1
out_dir           <- "simdata"
template          <- "Stool"   # built-in template, or a fit from fit_SparseDOSSA2()
n_per_arm         <- 20
weeks             <- c(0, 2, 4, 8, 12)
median_read_depth <- 50000

# Treatment x time effects (zero at baseline, grow linearly to full size at last week)
n_abund_only <- 10   # features with an abundance effect only
n_prev_only  <- 10   # features with a prevalence effect only
n_both       <- 10   # features with both
# Background drift over time in BOTH arms (so time alone is not the treatment effect)
n_drift      <- 5

lfc_abund  <- 1.5    # log fold change in abundance at last week (treatment vs control)
logor_prev <- 2      # log odds ratio for prevalence at last week
lfc_drift  <- 1      # log fold change across the whole study, both arms

set.seed(seed)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## ---- metadata --------------------------------------------------------------
n_subj <- 2 * n_per_arm
subjects <- data.frame(
  subject_id = sprintf("S%03d", seq_len(n_subj)),
  arm        = rep(c("control", "treatment"), each = n_per_arm),
  age        = round(rnorm(n_subj, mean = 35, sd = 8)),
  sex        = sample(c("F", "M"), n_subj, replace = TRUE),
  stringsAsFactors = FALSE
)
meta <- merge(subjects, data.frame(week = weeks), by = NULL)   # subjects x timepoints
meta <- meta[order(meta$subject_id, meta$week), ]
rownames(meta) <- NULL
meta$sample_id   <- sprintf("%s_wk%02d", meta$subject_id, meta$week)
meta$time_scaled <- meta$week / max(weeks)                     # 0..1
meta$trt_time    <- ifelse(meta$arm == "treatment", meta$time_scaled, 0)
meta <- meta[, c("sample_id", "subject_id", "arm", "week", "time_scaled",
                 "trt_time", "age", "sex")]

# Numeric matrix SparseDOSSA2 spikes against (age/sex are left as unspiked covariates)
meta_mat <- as.matrix(meta[, c("time_scaled", "trt_time")])

## ---- pick features to spike ------------------------------------------------
# Quick null run to find features that are neither always present nor vanishingly
# rare, so both prevalence and abundance effects are detectable.
null_sim <- SparseDOSSA2(template = template, n_sample = 500, new_features = FALSE,
                         spike_metadata = "none",
                         median_read_depth = median_read_depth, verbose = FALSE)
cnt0   <- null_sim$simulated_data
prev0  <- rowMeans(cnt0 > 0)
relab0 <- rowMeans(sweep(cnt0, 2, colSums(cnt0), "/"))

pool   <- names(prev0)[prev0 >= 0.3 & prev0 <= 0.9 & relab0 > 1e-4]
n_need <- n_abund_only + n_prev_only + n_both + n_drift
if (length(pool) < n_need) {
  stop("Only ", length(pool), " candidate features but ", n_need,
       " requested; lower the n_* settings or relax the pool filter.")
}
picked <- sample(pool, n_need)
grp <- factor(rep(c("abundance", "prevalence", "both", "drift"),
                  times = c(n_abund_only, n_prev_only, n_both, n_drift)),
              levels = c("abundance", "prevalence", "both", "drift"))
fg <- split(picked, grp)

sgn <- function(n) rep(c(1, -1), length.out = n)   # alternate up / down
mk_rows <- function(features, variable, property, effect) {
  if (length(features) == 0) return(NULL)
  data.frame(metadata_datum     = match(variable, colnames(meta_mat)),
             feature_spiked     = features,
             associated_property = property,
             effect_size        = effect,
             stringsAsFactors   = FALSE)
}

spike_df <- rbind(
  mk_rows(fg$abundance,  "trt_time",    "abundance",  lfc_abund  * sgn(length(fg$abundance))),
  mk_rows(fg$prevalence, "trt_time",    "prevalence", logor_prev * sgn(length(fg$prevalence))),
  mk_rows(fg$both,       "trt_time",    "abundance",  lfc_abund  * sgn(length(fg$both))),
  mk_rows(fg$both,       "trt_time",    "prevalence", logor_prev * sgn(length(fg$both))),
  mk_rows(fg$drift,      "time_scaled", "abundance",  lfc_drift  * sgn(length(fg$drift)))
)

truth <- data.frame(
  feature           = spike_df$feature_spiked,
  metadata_variable = colnames(meta_mat)[spike_df$metadata_datum],
  property          = spike_df$associated_property,
  effect_size       = spike_df$effect_size,
  null_prevalence   = round(prev0[spike_df$feature_spiked], 3),
  null_mean_relab   = signif(relab0[spike_df$feature_spiked], 3),
  stringsAsFactors  = FALSE
)

## ---- simulate --------------------------------------------------------------
sim <- SparseDOSSA2(template = template,
                    n_sample = nrow(meta),
                    new_features = FALSE,
                    spike_metadata = spike_df,
                    metadata_matrix = meta_mat,
                    median_read_depth = median_read_depth,
                    verbose = TRUE)

counts <- sim$simulated_data                 # features x samples
stopifnot(ncol(counts) == nrow(meta))
colnames(counts) <- meta$sample_id           # samples are in metadata row order
relab <- sweep(counts, 2, colSums(counts), "/")

## ---- write -----------------------------------------------------------------
wtsv <- function(x, file) write.table(x, file.path(out_dir, file), sep = "\t",
                                      quote = FALSE, row.names = FALSE)
wtsv(data.frame(feature = rownames(counts), counts, check.names = FALSE), "counts.tsv")
wtsv(data.frame(feature = rownames(relab),  relab,  check.names = FALSE), "relabundance.tsv")
wtsv(meta,  "metadata.tsv")
wtsv(truth, "spiked_truth.tsv")

## ---- sanity check: do the effects point the way the truth table says? -------
last <- max(weeks)
arm_stat <- function(f, arm, fun) {
  fun(counts[f, meta$arm == arm & meta$week == last], relab[f, meta$arm == arm & meta$week == last])
}
tt  <- truth[truth$metadata_variable == "trt_time", ]
chk <- data.frame(
  tt[, c("feature", "property", "effect_size")],
  prev_ctrl = sapply(tt$feature, arm_stat, arm = "control",   fun = function(c, r) mean(c > 0)),
  prev_trt  = sapply(tt$feature, arm_stat, arm = "treatment", fun = function(c, r) mean(c > 0)),
  rel_ctrl  = sapply(tt$feature, arm_stat, arm = "control",   fun = function(c, r) mean(r)),
  rel_trt   = sapply(tt$feature, arm_stat, arm = "treatment", fun = function(c, r) mean(r)),
  row.names = NULL
)
cat("\nWeek", last, "control vs treatment for treatment x time features:\n")
print(chk, digits = 3)
cat("\nWrote:", normalizePath(out_dir), "\n")
