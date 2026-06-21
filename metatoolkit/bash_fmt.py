#!/usr/bin/env python3
import sys
import shlex

def format_command(command_string):
    # shlex.split handles quotes correctly
    tokens = shlex.split(command_string)
    if not tokens:
        return

    cmd = tokens[0]
    print(f"{cmd} \\")

    i = 1
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("-") and token != "-":
            # Check if the next token is a value (doesn't start with '-')
            if i + 1 < len(tokens) and not (tokens[i+1].startswith("-") and tokens[i+1] != "-"):
                val = shlex.quote(tokens[i+1])
                print(f"  {token} {val} \\")
                i += 2
            else:
                print(f"  {token} \\")
                i += 1
        else:
            # Positional arguments or standalone "-"
            val = shlex.quote(token)
            print(f"  {val} \\")
            i += 1
            
    # Optional: Backspace the last backslash for cleaner copy-pasting
    # (Leaving it standard here to match your original script)

if __name__ == "__main__":
    # Accept input from stdin (piped) or arguments
    if not sys.stdin.isatty():
        input_cmd = sys.stdin.read().strip()
    else:
        input_cmd = " ".join(sys.argv[1:])
        
    format_command(input_cmd)
