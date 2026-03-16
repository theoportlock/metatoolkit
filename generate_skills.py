#!/usr/bin/env python
"""Generate skills.yaml from metatoolkit argparse scripts by parsing source."""

import re
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent / "metatoolkit"
OUTPUT_FILE = Path(__file__).parent / "skills.yaml"


def infer_type(arg_name, help_text=""):
    """Infer type from help text and argument name."""
    combined = (arg_name + " " + (help_text or "")).lower()
    
    if "tsv" in combined or "file" in combined or "path" in combined:
        return "tsv"
    if "csv" in combined:
        return "csv"
    if "nwk" in combined or "tree" in combined:
        return "nwk"
    if "json" in combined:
        return "json"
    if "formula" in combined:
        return "formula"
    if "output" in arg_name.lower() or "outfile" in arg_name.lower():
        return "tsv"
    if "input" in arg_name.lower():
        return "tsv"
    return "string"


def parse_script(script_path):
    """Parse argparse definitions from source code."""
    content = script_path.read_text()
    
    # Extract description from ArgumentParser
    desc_match = re.search(r'ArgumentParser\([^)]*description\s*=\s*["\']([^"\']+)', content)
    description = desc_match.group(1) if desc_match else ""
    
    if not description:
        # Try docstring
        docstring = re.search(r'"""([^"]+?)"""', content)
        if docstring:
            description = docstring.group(1).strip().split("\n")[0]
    
    if not description:
        description = f"{script_path.stem} tool"
    
    # Find all add_argument calls
    # Patterns:
    # add_argument("positional")
    # add_argument("-s", "--long", ...)
    # add_argument("--long", ...)
    
    inputs = []
    outputs = []
    params = []
    
    # Find all add_argument calls
    add_arg_calls = re.findall(
        r'add_argument\s*\(\s*["\']?([^"\']+)["\']?\s*,([^)]+)\)',
        content,
        re.MULTILINE
    )
    
    for arg_name_raw, args_block in add_arg_calls:
        arg_name_raw = arg_name_raw.strip().strip("\"'")
        args_block = args_block.strip()
        
        # Skip if this looks like a nested call
        if "add_argument" in args_block:
            continue
        
        # Extract help text
        help_match = re.search(r'help\s*=\s*["\']([^"\']*)', args_block)
        help_text = help_match.group(1) if help_match else ""
        
        # Check for default
        default_match = re.search(r'default\s*=\s*([^,\)]+)', args_block)
        default_val = default_match.group(1).strip() if default_match else None
        
        # Check for type
        type_match = re.search(r'type\s*=\s*(\w+)', args_block)
        arg_type = type_match.group(1) if type_match else None
        
        # Check for action
        action_match = re.search(r'action\s*=\s*(\w+)', args_block)
        action = action_match.group(1) if action_match else None
        
        # Check for nargs
        nargs_match = re.search(r'nargs\s*=\s*["\']?(\?)["\']?', args_block)
        nargs = nargs_match.group(1) if nargs_match else None
        
        # Determine if positional or optional
        is_positional = not arg_name_raw.startswith("-")
        
        # Check for choices/enum
        choices_match = re.search(r'choices\s*=\s*\[([^\]]+)\]', args_block)
        choices = None
        if choices_match:
            choices = [c.strip().strip("'\"") for c in choices_match.group(1).split(",")]
        
        # Determine the canonical name
        if is_positional:
            name = arg_name_raw
            inputs.append({
                "name": name,
                "type": infer_type(name, help_text),
                "required": True,
                "help": help_text
            })
        else:
            # Extract long name (after --)
            long_name = arg_name_raw
            if "," in arg_name_raw:
                parts = arg_name_raw.split(",")
                for p in parts:
                    if "--" in p:
                        long_name = p.strip()
                        break
            long_name = long_name.strip("-")
            
            # Check if this is output
            if long_name in ("output", "outfile", "out"):
                outputs.append({
                    "name": long_name,
                    "type": "tsv",
                    "help": help_text
                })
                continue
            
            # Boolean flag
            is_bool = action in ("store_true", "store_false")
            
            param = {
                "name": long_name,
                "help": help_text
            }
            
            if is_bool:
                param["type"] = "bool"
            elif choices:
                param["type"] = "choice"
                param["enum"] = choices
            elif arg_type:
                param["type"] = arg_type
            else:
                param["type"] = infer_type(long_name, help_text)
            
            if default_val and default_val not in ("None", "False", "True"):
                param["default"] = default_val
            
            params.append(param)
    
    return {
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
        "params": params
    }


def main():
    """Generate skills.yaml from metatoolkit scripts."""
    skills = {}
    
    for script in sorted(SCRIPT_DIR.glob("*.py")):
        if script.name.startswith("_"):
            continue
        if script.name == "generate_skills.py":
            continue
        
        try:
            skill = parse_script(script)
            
            if skill and (skill["inputs"] or skill["params"] or skill["outputs"]):
                skills[script.stem] = skill
                print(f"  {script.stem}: {len(skill['inputs'])} in, {len(skill['params'])} params")
            else:
                print(f"  {script.stem}: (empty)")
                
        except Exception as e:
            print(f"  {script.stem}: ERROR - {e}")
    
    # Write YAML
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Auto-generated from metatoolkit argparse scripts\n")
        f.write("# Run: python generate_skills.py\n\n")
        yaml.dump(skills, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"\nGenerated {OUTPUT_FILE} with {len(skills)} skills")


if __name__ == "__main__":
    main()
