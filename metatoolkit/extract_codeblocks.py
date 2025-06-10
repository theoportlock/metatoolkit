#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import argparse

# Map common language identifiers to file extensions
LANG_EXTENSION_MAP = {
    'python': 'py',
    'bash': 'sh',
    'shell': 'sh',
    'javascript': 'js',
    'html': 'html',
    'css': 'css',
    'java': 'java',
    'c++': 'cpp',
    'cpp': 'cpp',
    'c': 'c',
    'json': 'json',
    # add more mappings as needed
}

def extract_codeblocks(markdown_text):
    """
    Extracts code blocks from markdown text.
    Returns a list of tuples: (language, code).
    """
    # This regex matches code blocks:
    #  - It looks for triple backticks.
    #  - Optionally captures a language identifier (one or more word characters).
    #  - Then captures all text until the closing triple backticks.
    pattern = r'```(\w+)?\s*\n(.*?)\n```'
    return re.findall(pattern, markdown_text, re.DOTALL)

def main():
    parser = argparse.ArgumentParser(
        description="Extract code blocks from a Markdown file and save them to an output directory."
    )
    parser.add_argument("input_file", help="Path to the Markdown file to process.")
    parser.add_argument(
        "-o", "--output_dir", default=".", 
        help="Directory where code blocks will be saved (default: current directory)."
    )
    args = parser.parse_args()
    
    input_file = args.input_file
    output_dir = args.output_dir
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Read the entire markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    codeblocks = extract_codeblocks(content)
    if not codeblocks:
        print("No code blocks found in the markdown file.")
        sys.exit(0)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Process and write each code block
    for i, (lang, code) in enumerate(codeblocks):
        lang = lang.strip() if lang else ""
        # Determine the file extension
        extension = LANG_EXTENSION_MAP.get(lang.lower(), lang.lower() if lang else "txt")
        # For the first code block, use base_name.extension; for others, append a number
        if i == 0:
            out_filename = f"{base_name}.{extension}"
        else:
            out_filename = f"{base_name}{i}.{extension}"
        out_filepath = os.path.join(output_dir, out_filename)
        with open(out_filepath, 'w', encoding='utf-8') as out_file:
            out_file.write(code.strip() + "\n")
        print(f"Saved code block {i+1} to {out_filepath}")

if __name__ == "__main__":
    main()

