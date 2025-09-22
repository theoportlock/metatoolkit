#!/bin/bash
# Usage: ./run_ollama.sh "Prompt text" myfile.txt output.log

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <prompt> <input_file> <output_log>"
    exit 1
fi

# Define variables for clarity and ease of use
PROMPT="$1"
INPUT_FILE="$2"
OUTPUT_LOG="$3"
MODEL_NAME="my-llama2"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# Run the model
echo "Sending prompt + file to Ollama..."
(echo "$PROMPT"; cat "$INPUT_FILE") | ollama run $MODEL_NAME | tee "$OUTPUT_LOG"

