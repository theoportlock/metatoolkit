using FlashWeave
using CSV
using DataFrames
using Statistics

# 1. Define Paths
data_path = "results/timepoint/PCV2/species_short.tsv"
meta_data_path = "results/work/covmeta_preg.tsv"

# 2. Load Data
# Using normalizenames=true as per the CSV.jl documentation
species_df = CSV.read(data_path, DataFrame, delim='\t', normalizenames=true)
meta_df = CSV.read(meta_data_path, DataFrame, delim='\t', normalizenames=true)

# 3. Filter to common IDs and Sort
common_ids = intersect(species_df[:, 1], meta_df[:, 1])

species_filtered = filter(row -> row[1] in common_ids, species_df)
meta_filtered = filter(row -> row[1] in common_ids, meta_df)

sort!(species_filtered, [1])
sort!(meta_filtered, [1])

# 4. Apply Prevalence Threshold (0.2)
# We calculate prevalence only on the species columns (2 to end)
species_only = species_filtered[:, 2:end]
n_samples = size(species_only, 1)
threshold = 0.2

# Identify columns where the fraction of non-zero entries >= 0.2
kept_cols = [sum(col .> 0) / n_samples >= threshold for col in eachcol(species_only)]

# Subset the species data
species_final = species_only[:, kept_cols]
meta_final = meta_filtered[:, 2:end]

println("Prevalence filtering complete:")
println("- Original species count: $(size(species_only, 2))")
println("- Species remaining (>20% prevalence): $(size(species_final, 2))")
println("- Total samples aligned: $n_samples")

# 5. Save temporary aligned files
CSV.write("species_aligned.tmp.tsv", species_final, delim='\t')
CSV.write("meta_aligned.tmp.tsv", meta_final, delim='\t')

# 6. Run FlashWeave
netw_results = learn_network("species_aligned.tmp.tsv", "meta_aligned.tmp.tsv",
                             sensitive=true,
                             heterogeneous=false)

# 7. Save results
output_path = "results/timepoint/PCV2/network_results.gml"
save_network(output_path, netw_results)

println("Success: Network saved to $output_path")
