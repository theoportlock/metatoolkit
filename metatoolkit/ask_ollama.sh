#!/usr/bin/env bash
# Usage:
#   ./run_ollama.sh "Prompt text" file1 [file2 ...] -o output.log [-m model]

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <prompt> <file1> [file2 ...] -o <output_log> [-m model]"
    exit 1
fi

PROMPT="$1"
shift

MODEL_NAME="llama3.1"
OUTPUT_LOG=""

# Parse flags after positional files
INPUT_FILES=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            OUTPUT_LOG="$2"
            shift 2
            ;;
        -m)
            MODEL_NAME="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            INPUT_FILES+=("$1")
            shift
            ;;
    esac
done

if [ -z "$OUTPUT_LOG" ]; then
    echo "Error: -o <output_log> is required" >&2
    exit 1
fi

if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "Error: At least one input file must be provided" >&2
    exit 1
fi

# Safety parameters
CHARS_PER_TOKEN=4
HEADROOM_FRAC=0.9
MAX_LINES=500

# Check files
for f in "${INPUT_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: Input file '$f' not found" >&2
        exit 1
    fi
done

# Get model context length
CONTEXT_TOKENS=$(ollama show "$MODEL_NAME" \
    | awk '/context length/ {print $3}')

if [ -z "$CONTEXT_TOKENS" ]; then
    echo "Error: Could not determine context length for model '$MODEL_NAME'" >&2
    exit 1
fi

MAX_TOKENS=$(awk -v c="$CONTEXT_TOKENS" -v h="$HEADROOM_FRAC" \
    'BEGIN {printf "%.0f", c*h}')
MAX_CHARS=$(( MAX_TOKENS * CHARS_PER_TOKEN ))

# Count characters
PROMPT_CHARS=$(printf "%s" "$PROMPT" | wc -c)

FILES_CHARS=0
for f in "${INPUT_FILES[@]}"; do
    FILE_CHARS=$(head -n "$MAX_LINES" "$f" | wc -c)
    FILES_CHARS=$(( FILES_CHARS + FILE_CHARS ))
done

TOTAL_CHARS=$(( PROMPT_CHARS + FILES_CHARS ))

# Enforce limit
if [ "$TOTAL_CHARS" -gt "$MAX_CHARS" ]; then
    echo "Error: Prompt + input exceeds context limit"
    echo "  Model:              $MODEL_NAME"
    echo "  Context tokens:     $CONTEXT_TOKENS"
    echo "  Token budget:       $MAX_TOKENS"
    echo "  Max characters:     $MAX_CHARS"
    echo "  Actual characters:"
    echo "    Prompt:           $PROMPT_CHARS"
    echo "    Files (head):     $FILES_CHARS"
    echo "    Total:            $TOTAL_CHARS"
    exit 2
fi

echo "Context check passed:"
echo "  Characters used: $TOTAL_CHARS / $MAX_CHARS"
echo "  Files included:"
for f in "${INPUT_FILES[@]}"; do
    echo "    - $f (first $MAX_LINES lines)"
done

# Build input stream
{
    printf "%s\n\n" "$PROMPT"

    for f in "${INPUT_FILES[@]}"; do
        printf "===== BEGIN FILE: %s =====\n" "$f"
        head -n "$MAX_LINES" "$f"
        printf "\n===== END FILE: %s =====\n\n" "$f"
    done
} | ollama run --hidethinking "$MODEL_NAME" | tee "$OUTPUT_LOG"

