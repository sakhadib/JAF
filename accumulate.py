#!/usr/bin/env python3
"""
Accumulate all JSON results from model directories into a single CSV file.
Intelligently extracts data and adds model name for tracking.
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Any

def get_model_dirs(results_dir: str = "results") -> List[tuple]:
    """Get all model directories and their names."""
    model_dirs = []
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return model_dirs
    
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            model_dirs.append((item, item_path))
    
    return sorted(model_dirs)

def extract_json_data(json_file: str) -> Dict[str, Any]:
    """Extract data from a single JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None

def collect_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Collect all JSON results from all model directories."""
    all_results = []
    model_dirs = get_model_dirs(results_dir)
    
    print(f"Found {len(model_dirs)} model directories\n")
    
    for model_name, model_path in model_dirs:
        print(f"Processing: {model_name}")
        json_files = sorted([f for f in os.listdir(model_path) if f.endswith('.json')])
        
        for json_file in json_files:
            json_path = os.path.join(model_path, json_file)
            data = extract_json_data(json_path)
            
            if data:
                # Add model name to the data
                data['model'] = model_name
                all_results.append(data)
        
        print(f"  ✓ Loaded {len(json_files)} files")
    
    return all_results

def get_all_keys(results: List[Dict[str, Any]]) -> List[str]:
    """Get all unique keys from all results."""
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Reorder keys: model first, then prompt_id, then others
    keys = ['model', 'prompt_id']
    other_keys = sorted([k for k in all_keys if k not in keys])
    return keys + other_keys

def save_to_csv(results: List[Dict[str, Any]], output_file: str = "accumulated_results.csv"):
    """Save all results to a CSV file."""
    if not results:
        print("No results to save!")
        return
    
    keys = get_all_keys(results)
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            
            for result in results:
                # Create a row with all keys, filling missing values with empty strings
                row = {}
                for key in keys:
                    value = result.get(key, '')
                    # Normalize newlines and escape characters in text fields
                    if isinstance(value, str):
                        # Replace newlines and carriage returns with space
                        value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        # Replace multiple spaces with single space
                        value = ' '.join(value.split())
                    row[key] = value
                writer.writerow(row)
        
        print(f"\n✓ Saved {len(results)} results to {output_file}")
        print(f"Columns: {len(keys)}")
        print(f"Rows: {len(results)}")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")

def main():
    print("=" * 60)
    print("Accumulating all JSON results into CSV")
    print("=" * 60 + "\n")
    
    # Collect all results
    results = collect_all_results()
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nTotal results collected: {len(results)}")
    
    # Save to CSV
    save_to_csv(results, "accumulated_results.csv")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
