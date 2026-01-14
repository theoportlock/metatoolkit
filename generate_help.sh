#!/usr/bin/env bash

#set -euo pipefail

usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_FILE"
    echo "  -i   Directory containing executables"
    echo "  -o   Output file (txt/md/doc)"
    exit 1
}

INPUT_DIR=""
OUTPUT_FILE=""

while getopts "i:o:" opt; do
    case "$opt" in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_FILE" ]]; then
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: $INPUT_DIR is not a directory"
    exit 1
fi

# Clear output file
: > "$OUTPUT_FILE"

for cmd in "$INPUT_DIR"/*; do
    if [[ -x "$cmd" && ! -d "$cmd" ]]; then
        name="$(basename "$cmd")"

        {
            echo "### $name"
            echo '```'
            # Try -h, fallback to --help
            if ! "$cmd" -h > /dev/null 2>&1; then
                "$cmd" --help 2>&1
            else
                "$cmd" -h 2>&1
            fi
            echo '```'
            echo
        } >> "$OUTPUT_FILE"

    fi
done

echo "Documentation written to $OUTPUT_FILE"

