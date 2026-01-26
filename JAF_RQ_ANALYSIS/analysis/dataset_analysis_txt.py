"""
Dataset Analysis Script
Exports a comprehensive text file describing the dataset structure and contents.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

def get_column_stats(series, col_name):
    """Get statistics for a single column."""
    stats = []
    stats.append(f"    Column: {col_name}")
    stats.append(f"      - Dtype: {series.dtype}")
    stats.append(f"      - Non-null count: {series.notna().sum()} / {len(series)}")
    stats.append(f"      - Null count: {series.isna().sum()}")
    
    # Check if column contains vectors/arrays first (before trying nunique)
    first_valid = None
    if series.dtype == 'object' and series.notna().any():
        first_valid = series.dropna().iloc[0]
        if isinstance(first_valid, (list, np.ndarray)):
            stats.append(f"      - Contains: VECTOR/ARRAY data")
            stats.append(f"      - Vector length: {len(first_valid)}")
            stats.append(f"      - Vector element type: {type(first_valid[0]).__name__ if len(first_valid) > 0 else 'N/A'}")
            # Calculate vector statistics
            all_vectors = series.dropna().tolist()
            if len(all_vectors) > 0:
                sample_vec = np.array(first_valid)
                stats.append(f"      - Sample vector stats - Min: {sample_vec.min():.6f}, Max: {sample_vec.max():.6f}, Mean: {sample_vec.mean():.6f}")
            # Show sample of first few elements
            sample_elements = list(first_valid[:5]) if len(first_valid) > 5 else list(first_valid)
            stats.append(f"      - Sample elements (first 5): {[round(x, 6) for x in sample_elements]}")
            return stats, True  # Return flag indicating vector column
    
    # Safe to calculate unique values for non-vector columns
    try:
        stats.append(f"      - Unique values: {series.nunique()}")
    except TypeError:
        stats.append(f"      - Unique values: N/A (unhashable type)")
    
    # Check if column contains strings
    if series.dtype == 'object' and first_valid is not None:
        if isinstance(first_valid, str):
            avg_len = series.dropna().str.len().mean()
            max_len = series.dropna().str.len().max()
            min_len = series.dropna().str.len().min()
            stats.append(f"      - String length - Min: {min_len}, Max: {max_len}, Avg: {avg_len:.1f}")
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(series):
        stats.append(f"      - Min: {series.min()}")
        stats.append(f"      - Max: {series.max()}")
        stats.append(f"      - Mean: {series.mean():.4f}")
        stats.append(f"      - Std: {series.std():.4f}")
        stats.append(f"      - Median: {series.median():.4f}")
    
    # Show sample values (for non-vector columns)
    sample_values = series.dropna().head(3).tolist()
    # Truncate long string values for display
    display_samples = []
    for val in sample_values:
        if isinstance(val, str) and len(val) > 100:
            display_samples.append(val[:100] + "...")
        else:
            display_samples.append(val)
    stats.append(f"      - Sample values: {display_samples}")
    
    return stats, False


def analyze_dataframe(df, name):
    """Analyze a dataframe and return description lines."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"DATASET: {name}")
    lines.append(f"{'='*80}")
    lines.append(f"\nBasic Info:")
    lines.append(f"  - Rows: {len(df)}")
    lines.append(f"  - Columns: {len(df.columns)}")
    lines.append(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    lines.append(f"\nColumn List:")
    for i, col in enumerate(df.columns, 1):
        lines.append(f"  {i}. {col} ({df[col].dtype})")
    
    lines.append(f"\nDetailed Column Analysis:")
    lines.append("-" * 60)
    
    vector_columns = []
    for col in df.columns:
        col_stats, is_vector = get_column_stats(df[col], col)
        lines.extend(col_stats)
        lines.append("")
        if is_vector:
            vector_columns.append(col)
    
    if vector_columns:
        lines.append(f"\n⚠️  VECTOR COLUMNS DETECTED: {vector_columns}")
        lines.append("    (These columns contain high-dimensional embedding vectors)")
    
    # Value counts for categorical columns with few unique values
    lines.append(f"\nValue Distributions (for columns with ≤20 unique values):")
    lines.append("-" * 60)
    for col in df.columns:
        if col in vector_columns:
            continue
        if df[col].nunique() <= 20 and df[col].nunique() > 0:
            lines.append(f"\n  {col}:")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                display_val = val if not isinstance(val, str) or len(str(val)) <= 50 else str(val)[:50] + "..."
                lines.append(f"    - {display_val}: {count} ({pct:.1f}%)")
    
    return lines


def main():
    # Setup paths
    data_dir = Path("data")
    output_file = Path("dataset_analysis_report.txt")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    # Analyze CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        report_lines.append("\n\n" + "#" * 80)
        report_lines.append("# CSV FILES")
        report_lines.append("#" * 80)
        
        for csv_file in csv_files:
            print(f"Analyzing: {csv_file}")
            df = pd.read_csv(csv_file)
            report_lines.extend(analyze_dataframe(df, csv_file.name))
            
            # Show first few rows (text representation)
            report_lines.append(f"\nFirst 5 rows preview:")
            report_lines.append("-" * 60)
            # Limit column width for display
            pd.set_option('display.max_colwidth', 50)
            report_lines.append(df.head().to_string())
    
    # Analyze Parquet files
    parquet_dir = data_dir / "parquet"
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            report_lines.append("\n\n" + "#" * 80)
            report_lines.append("# PARQUET FILES (Embedding Files)")
            report_lines.append("#" * 80)
            report_lines.append("\n⚠️  NOTE: Parquet files contain embedding vectors.")
            report_lines.append("    Vector data is summarized (not fully displayed) to keep report readable.")
            
            for parquet_file in parquet_files:
                print(f"Analyzing: {parquet_file}")
                df = pd.read_parquet(parquet_file)
                report_lines.extend(analyze_dataframe(df, parquet_file.name))
                
                # For parquet files, show structure but not full data
                report_lines.append(f"\nSchema preview (first 3 rows, vectors truncated):")
                report_lines.append("-" * 60)
                
                # Create a display version with truncated vectors
                display_df = df.head(3).copy()
                for col in display_df.columns:
                    first_val = display_df[col].iloc[0] if len(display_df) > 0 else None
                    if isinstance(first_val, (list, np.ndarray)):
                        display_df[col] = display_df[col].apply(
                            lambda x: f"[vector: {len(x)} dims]" if isinstance(x, (list, np.ndarray)) else x
                        )
                report_lines.append(display_df.to_string())
    
    # Summary section
    report_lines.append("\n\n" + "#" * 80)
    report_lines.append("# SUMMARY")
    report_lines.append("#" * 80)
    report_lines.append(f"\nTotal CSV files analyzed: {len(csv_files) if csv_files else 0}")
    report_lines.append(f"Total Parquet files analyzed: {len(parquet_files) if parquet_dir.exists() else 0}")
    
    if parquet_dir.exists() and parquet_files:
        report_lines.append("\nEmbedding Models Found:")
        for pf in parquet_files:
            model_name = pf.stem.replace("_", "/", 1)  # Convert back to model name format
            report_lines.append(f"  - {model_name}")
    
    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ Report saved to: {output_file}")
    print(f"   Report size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
