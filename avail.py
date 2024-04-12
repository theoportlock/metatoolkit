#!/usr/bin/env bash
printf '\e[1;34m%-6s\e[m' "Datasets: "
basename -a $(ls ../results/*.tsv) | xargs
printf '\n'

printf '\e[1;34m%-6s\e[m' "Analysis: "
basename -a $(ls ~/omicsbuilder/*) | xargs
