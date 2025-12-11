#!/bin/bash
source env.sh
set -e

#input=results/table1/work/all_data_merged.tsv
#output=results/table1/work/stats
input=results/table2/work/data_merged.tsv
output=results/table2/work/MAM_recovery_stats
input=$1
output=$2

rm -rf $output
mkdir -p $output

filter.py $input \
	--dtype bool \
	-o $output/categories.tsv

filter.py $input \
	--dtype number \
	-o $output/numbers.tsv

corr.py \
	$output/numbers.tsv \
	--dropna \
	-o $output/corr.tsv

mannwhitneyu.py \
	$output/numbers.tsv \
	$output/categories.tsv \
	--dropna \
	-o $output/mwu.tsv

fisher.py \
	$output/categories.tsv \
	--dropna \
	-o $output/fisher.tsv

merge.py \
	$output/corr.tsv \
	$output/mwu.tsv \
	$output/fisher.tsv \
	-a \
	--add-filename \
	--filename-format base \
	-o $output/merged_stats.tsv
