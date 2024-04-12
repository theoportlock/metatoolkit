#!/usr/bin/env Rscript

# Install and load required packages
#install.packages("vegan")
library(vegan)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
data_file <- '../results/sleepIDRecovery0.0vs1.0.tsv
# CONTINUE WITH THIS AFTER I HAVE FIXED THE EV SCRIPTS
data_file <- args[1]
group_variable <- args[2]
distance_variable <- args[3]
results_directory <- '../results/' + 'adonis'

# Load data from TSV file
data <- read.table(data_file, sep="\t", header=TRUE)

# Perform PERMANOVA analysis
permanova_result <- adonis(formula(paste(distance_variable, "~", group_variable)), data=data, permutations=999)

# Print the PERMANOVA results
print(permanova_result)

# Save results to specified directory
results_file <- paste(results_directory, "/permanova_results.txt", sep="")
write.table(permanova_result, file=results_file, quote=FALSE)

