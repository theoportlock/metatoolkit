#!/usr/bin/env python

import argparse
import os
import requests
import sys

def parse_args():
    """Handles command-line arguments."""
    parser = argparse.ArgumentParser(description="Download a PDB structure file.")
    
    parser.add_argument(
        "-i", "--id", 
        required=True, 
        help="The 4-character PDB ID (e.g., 7PMP)"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="The output file path (e.g., results/7pmp.pdb)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    pdb_id = args.id.upper()
    output_path = args.output

    # 1. Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 2. Construct the URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    print(f"Downloading {pdb_id} from RCSB...")
    
    try:
        response = requests.get(url)
        # Check if the request was successful (HTTP 200)
        response.raise_for_status()
        
        # 3. Save the file
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        print(f"Successfully saved to: {output_path}")

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"Error: PDB ID '{pdb_id}' not found.")
        else:
            print(f"HTTP Error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
