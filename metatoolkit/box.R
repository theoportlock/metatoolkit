#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tibble)
  library(rlang)    # for sym()
})

# ---- Parse arguments ----
parser <- ArgumentParser(description = "Produces a Boxplot of a given dataset")
parser$add_argument("subject", help = "Path to dataset file or subject name")
parser$add_argument("-x", help = "Column name for x-axis")
parser$add_argument("-y", help = "Column name for y-axis")
parser$add_argument("--hue", help = "Column name for hue grouping")
parser$add_argument("--logy", action = "store_true", help = "Set y-axis to log scale")
parser$add_argument("--show", action = "store_true", help = "Display the plot window")
parser$add_argument("--figsize", default = "2,2", help = "Figure size as width,height (in inches)")
parser$add_argument("-o", "--output", help = "Output filename without extension")
parser$add_argument("--meta", nargs = "+", help = "Path(s) to metadata file(s) to inner-join")
parser$add_argument("--rc", help = "Path to ggplot2 theme file (RDS with theme)")

args <- parser$parse_args()

# ---- Load data ----
load_data <- function(path_or_name) {
  if (file.exists(path_or_name)) {
    df <- read_tsv(path_or_name)
  } else {
    fname <- file.path("results", paste0(path_or_name, ".tsv"))
    df <- read_tsv(fname)
  }
  df <- as.data.frame(df)
  rownames(df) <- df[[1]]
  df <- df[,-1, drop = FALSE]
  return(df)
}

# ---- Merge metadata ----
merge_meta <- function(df, meta_paths) {
  for (mpath in meta_paths) {
    mdf <- read_delim(mpath, delim = NULL, col_types = cols())
    mdf <- as.data.frame(mdf)
    rownames(mdf) <- mdf[[1]]
    mdf <- mdf[,-1, drop = FALSE]
    df <- inner_join(
      df %>% rownames_to_column("id"),
      mdf %>% rownames_to_column("id"),
      by = "id"
    ) %>% column_to_rownames("id")
  }
  return(df)
}

# ---- Plot function ----
plot_box <- function(df, x, y, hue, figsize) {
  df <- rownames_to_column(df, "index")

  x_col <- if (is.null(x)) colnames(df)[2] else x
  y_col <- if (is.null(y)) colnames(df)[3] else y

  # add small pseudocount to avoid log10(0)
  df[[y_col]][df[[y_col]] == 0] <- 1e-6

  # build base aesthetic
  if (is.null(hue)) {
    aes_mapping <- aes(x = !!sym(x_col), y = !!sym(y_col))
  } else {
    aes_mapping <- aes(x = !!sym(x_col), y = !!sym(y_col),
                       fill = !!sym(hue), color = !!sym(hue))
  }

  p <- ggplot(df, aes_mapping) +
    geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.7) +
    geom_jitter(size = 1.2, width = 0.2, alpha = 0.8) +
    theme_classic(base_size = 10) +
    theme(
      panel.background = element_rect(fill = "#F5F0DC", colour = NA),
      axis.text  = element_text(color = "black"),
      axis.title = element_text(color = "black")
    )

  # Add colors only if hue is defined
  if (!is.null(hue)) {
    p <- p +
      scale_fill_manual(values = c("#C0DFA1", "#7FA960")) +
      scale_color_manual(values = c("#C0DFA1", "#7FA960"))
  }

  return(p)
}

# ---- Main ----
df <- load_data(args$subject)
print(df)

if (!is.null(args$meta)) {
  df <- merge_meta(df, args$meta)
}

# load theme if provided
if (!is.null(args$rc)) {
  theme_custom <- readRDS(args$rc)
  if (inherits(theme_custom, "theme")) {
    theme_set(theme_custom)
  }
}

# parse figsize
figsize <- as.numeric(strsplit(args$figsize, ",")[[1]])

# plot
p <- plot_box(df, args$x, args$y, args$hue, figsize)
if (args$logy) {
  p <- p + scale_y_log10()
}

# save
outfile <- if (!is.null(args$output)) {
  args$output
} else {
  file.path("results", paste0(tools::file_path_sans_ext(basename(args$subject)), "_box.svg"))
}
dir.create(dirname(outfile), showWarnings = FALSE, recursive = TRUE)

ggsave(outfile, plot = p, width = figsize[1], height = figsize[2])

if (args$show) {
  print(p)
}

