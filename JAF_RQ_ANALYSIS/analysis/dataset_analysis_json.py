"""
Dataset Analysis - Structured JSON Export for LLM Understanding
Exports a comprehensive JSON schema describing the dataset structure.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


def analyze_column(series, col_name, categorical_threshold=50):
    """Analyze a single column and return structured info."""
    col_info = {
        "name": col_name,
        "dtype": str(series.dtype),
        "total_count": len(series),
        "non_null_count": int(series.notna().sum()),
        "null_count": int(series.isna().sum()),
        "null_percentage": round(series.isna().sum() / len(series) * 100, 2)
    }
    
    # Check for vector/array data first
    if series.dtype == 'object' and series.notna().any():
        first_valid = series.dropna().iloc[0]
        if isinstance(first_valid, (list, np.ndarray)):
            col_info["column_type"] = "embedding_vector"
            col_info["vector_dimensions"] = len(first_valid)
            col_info["element_dtype"] = type(first_valid[0]).__name__ if len(first_valid) > 0 else "unknown"
            sample_vec = np.array(first_valid)
            col_info["vector_stats"] = {
                "min": round(float(sample_vec.min()), 6),
                "max": round(float(sample_vec.max()), 6),
                "mean": round(float(sample_vec.mean()), 6)
            }
            col_info["note"] = "High-dimensional embedding vector - not suitable for direct analysis, use for similarity/clustering"
            return col_info
    
    # Calculate unique values for non-vector columns
    try:
        unique_count = series.nunique()
        col_info["unique_count"] = int(unique_count)
    except TypeError:
        col_info["unique_count"] = "N/A (unhashable)"
        return col_info
    
    # Determine if categorical (unique < threshold)
    is_categorical = unique_count < categorical_threshold
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(series):
        col_info["column_type"] = "categorical_numeric" if is_categorical else "continuous_numeric"
        col_info["statistics"] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "median": float(series.median())
        }
        if is_categorical:
            value_counts = series.value_counts().to_dict()
            col_info["value_distribution"] = {str(k): int(v) for k, v in value_counts.items()}
    
    # For string/object columns
    elif series.dtype == 'object':
        if is_categorical:
            col_info["column_type"] = "categorical_string"
            value_counts = series.value_counts().to_dict()
            col_info["value_distribution"] = {str(k): int(v) for k, v in value_counts.items()}
        else:
            col_info["column_type"] = "text"
            str_lengths = series.dropna().str.len()
            col_info["string_stats"] = {
                "min_length": int(str_lengths.min()),
                "max_length": int(str_lengths.max()),
                "avg_length": round(float(str_lengths.mean()), 1)
            }
            # Sample values (truncated)
            samples = series.dropna().head(2).tolist()
            col_info["sample_values"] = [s[:100] + "..." if len(s) > 100 else s for s in samples]
    
    return col_info


def analyze_dataframe(df, name):
    """Analyze a dataframe and return structured info."""
    df_info = {
        "file_name": name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "columns": []
    }
    
    for col in df.columns:
        col_info = analyze_column(df[col], col)
        df_info["columns"].append(col_info)
    
    return df_info


def main():
    data_dir = Path("data")
    output_file = Path("dataset_schema.json")
    
    # Build the structured schema
    schema = {
        "dataset_description": {
            "title": "Bangladeshi Folklore Story Generation & Evaluation Dataset",
            "description": "This dataset contains AI-generated folklore stories from various Bangladeshi cultural groups, along with human evaluations and text embeddings for semantic analysis.",
            "domain": "Cultural AI / NLP / Folklore Generation",
            "language": "Bengali (Bangla)",
            "generated_at": datetime.now().isoformat()
        },
        "data_files": {
            "evaluation_data": None,
            "embedding_files": []
        },
        "relationships": {
            "linking_keys": {
                "primary_key_combination": ["model", "culture"],
                "description": "The combination of (model, culture) serves as the linking key between evaluated_stories.csv and all parquet embedding files. Each parquet file contains the same 599 stories with identical (model, culture) pairs but different embedding representations.",
                "join_type": "one-to-one per embedding model"
            },
            "file_relationships": [
                {
                    "from_file": "evaluated_stories.csv",
                    "to_files": "*.parquet files",
                    "relationship": "Each story in evaluated_stories.csv has corresponding embeddings in each parquet file",
                    "link_columns": ["model", "culture", "story"]
                }
            ]
        },
        "semantic_overview": {
            "data_generation_pipeline": [
                "1. Prompts were created for 12 Bangladeshi cultures × 10 story types = 120 unique prompts",
                "2. 5 different LLMs generated stories from these prompts",
                "3. Stories were evaluated by humans on 4 quality dimensions (0-5 scale)",
                "4. Story text was embedded using 5 different embedding models"
            ],
            "key_dimensions": {
                "cultures": {
                    "count": 12,
                    "includes": "Bengali, Chakma, Garo (Mandi), Hajong, Khasi, Manipuri (Meitei), Marma, Mro, Oraon (Kurukh), Rakhine, Santal, Tripura"
                },
                "generation_models": {
                    "count": 5,
                    "includes": "google_gemini-3-flash-preview, mistralai_mistral-large-2512, openai_gpt-5-mini, openai_gpt-5.1, qwen_qwen3-8b"
                },
                "story_types": {
                    "count": 10,
                    "includes": "Change and Continuity, Community Crisis, Everyday Life Narrative, Human–Nature Relationship, Moral Transgression and Consequence, Origin Story, Rite of Passage, Sacred or Forbidden Space, Trickster or Clever Figure, Wisdom of an Elder"
                },
                "embedding_models": {
                    "count": 5,
                    "includes": "google/gemini-embedding-001 (3072d), mistralai/mistral-embed-2312 (1024d), openai/text-embedding-3-large (3072d), openai/text-embedding-3-small (1536d), qwen/qwen3-embedding-8b (4096d)"
                }
            },
            "evaluation_metrics": {
                "scale": "0-5 (higher is better)",
                "dimensions": [
                    {"name": "Cultural Accuracy & Authenticity", "description": "How accurately the story reflects the target culture"},
                    {"name": "Contextual & Temporal Appropriateness", "description": "Whether the story fits traditional/historical context"},
                    {"name": "Narrative & Symbolic Coherence", "description": "Story structure and use of cultural symbols"},
                    {"name": "Linguistic & Expressive Appropriateness (Bangla)", "description": "Quality of Bengali language usage"}
                ]
            }
        },
        "usage_notes": {
            "for_analysis": [
                "Use evaluated_stories.csv for quality analysis across models and cultures",
                "Join with parquet files on (model, culture, story) for embedding-based analysis",
                "Embedding vectors can be used for clustering, similarity search, or as features"
            ],
            "categorical_columns": "Columns with <50 unique values are treated as categorical",
            "text_columns": "story, prompt, scenario contain long Bengali text suitable for NLP",
            "vector_columns": "scenario_embedding and story_embedding contain high-dimensional float vectors"
        }
    }
    
    # Analyze CSV file
    csv_file = data_dir / "evaluated_stories.csv"
    if csv_file.exists():
        print(f"Analyzing: {csv_file}")
        df = pd.read_csv(csv_file)
        schema["data_files"]["evaluation_data"] = analyze_dataframe(df, csv_file.name)
    
    # Analyze parquet files
    parquet_dir = data_dir / "parquet"
    if parquet_dir.exists():
        for parquet_file in sorted(parquet_dir.glob("*.parquet")):
            print(f"Analyzing: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            parquet_info = analyze_dataframe(df, parquet_file.name)
            
            # Add embedding model info
            embedding_model = df['embedding_model'].iloc[0] if 'embedding_model' in df.columns else "unknown"
            parquet_info["embedding_model"] = embedding_model
            
            # Get vector dimensions
            for col_info in parquet_info["columns"]:
                if col_info.get("column_type") == "embedding_vector":
                    parquet_info["vector_dimensions"] = col_info.get("vector_dimensions")
                    break
            
            schema["data_files"]["embedding_files"].append(parquet_info)
    
    # Add summary statistics
    schema["summary_statistics"] = {
        "total_stories": 599,
        "total_evaluation_records": 599,
        "embedding_files_count": len(schema["data_files"]["embedding_files"]),
        "evaluation_score_ranges": {
            "Cultural Accuracy & Authenticity": {"min": 0, "max": 5, "mean": 2.45},
            "Contextual & Temporal Appropriateness": {"min": 0, "max": 5, "mean": 2.17},
            "Narrative & Symbolic Coherence": {"min": 0, "max": 5, "mean": 1.99},
            "Linguistic & Expressive Appropriateness (Bangla)": {"min": 0, "max": 5, "mean": 1.44}
        }
    }
    
    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Schema saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Also create a compact LLM-ready summary
    llm_summary = {
        "dataset_name": "Bangladeshi Folklore Story Generation & Evaluation Dataset",
        "purpose": "Evaluate AI-generated cultural folklore stories across 12 Bangladeshi cultures",
        "total_stories": 599,
        "files": {
            "evaluated_stories.csv": {
                "role": "Main evaluation file with human ratings",
                "columns": {
                    "story_id": "Unique identifier (1-599)",
                    "culture": "Target culture (12 categories: Bengali, Chakma, Garo, Hajong, Khasi, Manipuri, Marma, Mro, Oraon, Rakhine, Santal, Tripura)",
                    "model": "LLM that generated the story (5 models)",
                    "story": "Generated folklore story in Bengali (500-700 words)",
                    "Cultural Accuracy & Authenticity": "Human rating 0-5",
                    "Contextual & Temporal Appropriateness": "Human rating 0-5",
                    "Narrative & Symbolic Coherence": "Human rating 0-5",
                    "Linguistic & Expressive Appropriateness (Bangla)": "Human rating 0-5"
                }
            },
            "parquet/*.parquet": {
                "role": "Embedding files (5 files, one per embedding model)",
                "key_columns": {
                    "model": "LLM that generated the story (LINK KEY)",
                    "culture": "Target culture (LINK KEY)",
                    "story": "Same story text as in CSV",
                    "story_type": "Narrative category (10 types)",
                    "region": "Geographic region in Bangladesh",
                    "scenario_embedding": "Vector embedding of the story prompt/scenario",
                    "story_embedding": "Vector embedding of the generated story"
                },
                "embedding_dimensions": {
                    "google_gemini-embedding-001": 3072,
                    "mistralai_mistral-embed-2312": 1024,
                    "openai_text-embedding-3-large": 3072,
                    "openai_text-embedding-3-small": 1536,
                    "qwen_qwen3-embedding-8b": 4096
                }
            }
        },
        "IMPORTANT_linking_key": "(model, culture) links evaluated_stories.csv to all parquet files - same 599 stories appear in each file with different embeddings"
    }
    
    llm_summary_file = Path("dataset_schema_compact.json")
    with open(llm_summary_file, "w", encoding="utf-8") as f:
        json.dump(llm_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Compact LLM summary saved to: {llm_summary_file}")


if __name__ == "__main__":
    main()
