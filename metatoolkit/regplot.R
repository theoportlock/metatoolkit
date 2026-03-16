#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(ggplot2)
  library(readr)
  library(dplyr)
})

parse_arguments <- function() {
  parser <- ArgumentParser(description = "Produces a regression plot of a dataset")

  parser$add_argument("subject", help="Path to dataset file or subject name")
  parser$add_argument("-o", "--output", help="Output svg file")
  parser$add_argument("-x", help="Column name for x-axis")
  parser$add_argument("-y", help="Column name for y-axis")
  parser$add_argument("--hue", help="Column name for colour grouping")
  parser$add_argument("--logy", action="store_true", help="Log scale y-axis")
  parser$add_argument("--logx", action="store_true", help="Log scale x-axis")
  parser$add_argument("--show", action="store_true", help="Display plot window")
  parser$add_argument("--figsize", default="2,2", help="Figure size width,height")

  parser$parse_args()
}

load_data <- function(subject) {

  if (file.exists(subject)) {
    df <- read_tsv(subject, show_col_types = FALSE)
  } else {
    path <- file.path("results", paste0(subject, ".tsv"))
    df <- read_tsv(path, show_col_types = FALSE)
  }

  df
}

plot_reg <- function(df, x=NULL, y=NULL, hue=NULL) {

  if (is.null(x)) x <- colnames(df)[1]
  if (is.null(y)) y <- colnames(df)[2]

  p <- ggplot(df, aes_string(x=x, y=y))

  if (!is.null(hue)) {
    p <- p +
      geom_point(aes_string(color=hue), size=0.5) +
      guides(color="none")
  } else {
    p <- p +
      geom_point(size=0.5, colour="black")
  }

  p <- p +
    geom_smooth(method="lm", se=FALSE, colour="red", linewidth=0.4) +
    theme_classic(base_size = 8) +
    theme(

      panel.grid.major = element_line(
        colour="grey85",
        linewidth=0.3
      ),

      panel.grid.minor = element_line(
        colour="grey92",
        linewidth=0.2
      ),

      axis.line = element_line(
        colour="black",
        linewidth=0.4
      ),

      axis.ticks = element_line(
        linewidth=0.4
      )
    )

  p
}

main <- function() {

  args <- parse_arguments()

  df <- load_data(args$subject)

  dims <- as.numeric(strsplit(args$figsize, ",")[[1]])
  width <- dims[1]
  height <- dims[2]

  p <- plot_reg(df, args$x, args$y, args$hue)

  if (args$logx) {
    p <- p + scale_x_log10()
  }

  if (args$logy) {
    p <- p + scale_y_log10()
  }

  outfile <- ifelse(
    is.null(args$output),
    paste0(args$subject, "_regplot.svg"),
    args$output
  )

  ggsave(
    outfile,
    p,
    width = width,
    height = height,
    units = "in"
  )

  if (args$show) {
    print(p)
  }
}

main()
