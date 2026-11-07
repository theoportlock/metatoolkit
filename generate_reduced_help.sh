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
            echo '```text'
            
            # Capture help output
            if ! "$cmd" -h > /dev/null 2>&1; then
                HELP_OUT=$("$cmd" --help 2>&1)
            else
                HELP_OUT=$("$cmd" -h 2>&1)
            fi

            # Pipe output through awk to drastically reduce LLM token count
            echo "$HELP_OUT" | awk '
            # 1. Skip the token-heavy "Usage:" block and its indented continuations
            /^[Uu]sage:/ { in_usage=1; next }
            in_usage {
                if (/^$/ || /^[ \t]+/) next
                in_usage=0
            }
            
            # 2. Skip standard help flags (LLMs already know what these do)
            /^[ \t]*-h,[ \t]*--help/ { next }
            
            # 3. Skip redundant section headers
            /^(positional arguments|options|Options|optional arguments):[ \t]*$/ { next }
            
            # 4. Compress 3+ spaces/tabs into 2 spaces to save whitespace tokens
            {
                gsub(/[ \t]{3,}/, "  ")
                print
            }' | grep -v '^[[:space:]]*$' # 5. Remove all empty lines

            echo '```'
            echo
        } >> "$OUTPUT_FILE"

    fi
done

echo "Documentation written to $OUTPUT_FILE"
