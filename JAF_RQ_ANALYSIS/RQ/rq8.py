"""
RQ8: Embedding Model Sensitivity Comparison
============================================
Question: Does higher dimensionality capture more nuance?
          Which embedding model best differentiates cultural narratives?

Method:
- For each of 5 embedding files, load story_embedding
- Filter by story_type, compute inter-culture distances
- Measure variance: Which embedding model shows highest distance variance?
- Compare dimensions: 1024 (mistral) vs 3072 (openai-large/gemini) vs 1536 (openai-small) vs 4096 (qwen)

Data Source: All 5 parquet files

Embedding Models:
- mistralai_mistral-embed-2312: 1024 dimensions
- openai_text-embedding-3-small: 1536 dimensions
- google_gemini-embedding-001: 3072 dimensions
- openai_text-embedding-3-large: 3072 dimensions
- qwen_qwen3-embedding-8b: 4096 dimensions
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, pdist, squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")


# Embedding model specifications
EMBEDDING_MODELS = {
    'mistralai_mistral-embed-2312': {'name': 'Mistral', 'dim': 1024},
    'openai_text-embedding-3-small': {'name': 'OpenAI-Small', 'dim': 1536},
    'google_gemini-embedding-001': {'name': 'Gemini', 'dim': 3072},
    'openai_text-embedding-3-large': {'name': 'OpenAI-Large', 'dim': 3072},
    'qwen_qwen3-embedding-8b': {'name': 'Qwen', 'dim': 4096}
}


def load_all_parquet_files():
    """Load all parquet files and return as dict."""
    parquet_dir = Path(__file__).parent.parent / "data" / "parquet"
    
    data = {}
    for filename, info in EMBEDDING_MODELS.items():
        filepath = parquet_dir / f"{filename}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            data[filename] = {
                'df': df,
                'name': info['name'],
                'dim': info['dim']
            }
            print(f"    Loaded {info['name']}: {len(df)} rows, {info['dim']} dims")
        else:
            print(f"    Warning: {filepath.name} not found")
    
    return data


def extract_embeddings(df):
    """Extract story embeddings as numpy array."""
    return np.array(df['story_embedding'].tolist())


def compute_inter_culture_distances(embeddings, cultures, metric='cosine'):
    """
    Compute pairwise distances between different cultures.
    Returns distances for pairs where culture_i != culture_j.
    """
    unique_cultures = np.unique(cultures)
    
    # Calculate centroid for each culture
    centroids = {}
    for culture in unique_cultures:
        mask = cultures == culture
        centroids[culture] = embeddings[mask].mean(axis=0)
    
    # Pairwise centroid distances
    distances = []
    for i, c1 in enumerate(unique_cultures):
        for j, c2 in enumerate(unique_cultures):
            if i < j:
                if metric == 'cosine':
                    dist = 1 - np.dot(centroids[c1], centroids[c2]) / (
                        np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]))
                else:
                    dist = np.linalg.norm(centroids[c1] - centroids[c2])
                distances.append({
                    'culture_1': c1,
                    'culture_2': c2,
                    'distance': dist
                })
    
    return pd.DataFrame(distances)


def compute_inter_culture_distances_by_story_type(embeddings, cultures, story_types, metric='cosine'):
    """
    Compute inter-culture distances for each story type separately.
    This measures how well the embedding differentiates cultures within the same story type.
    """
    unique_story_types = np.unique(story_types)
    unique_cultures = np.unique(cultures)
    
    all_distances = []
    
    for st in unique_story_types:
        st_mask = story_types == st
        st_embeddings = embeddings[st_mask]
        st_cultures = cultures[st_mask]
        
        # Calculate centroid for each culture within this story type
        centroids = {}
        for culture in unique_cultures:
            c_mask = st_cultures == culture
            if c_mask.sum() > 0:
                centroids[culture] = st_embeddings[c_mask].mean(axis=0)
        
        # Pairwise distances between cultures
        culture_list = list(centroids.keys())
        for i, c1 in enumerate(culture_list):
            for j, c2 in enumerate(culture_list):
                if i < j:
                    if metric == 'cosine':
                        dist = 1 - np.dot(centroids[c1], centroids[c2]) / (
                            np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]))
                    else:
                        dist = np.linalg.norm(centroids[c1] - centroids[c2])
                    
                    all_distances.append({
                        'story_type': st,
                        'culture_1': c1,
                        'culture_2': c2,
                        'distance': dist
                    })
    
    return pd.DataFrame(all_distances)


def compute_sensitivity_metrics(distances_df):
    """
    Compute sensitivity metrics for an embedding model.
    Higher variance = better differentiation = more sensitive to cultural nuances.
    """
    return {
        'mean_distance': distances_df['distance'].mean(),
        'std_distance': distances_df['distance'].std(),
        'variance': distances_df['distance'].var(),
        'min_distance': distances_df['distance'].min(),
        'max_distance': distances_df['distance'].max(),
        'range': distances_df['distance'].max() - distances_df['distance'].min(),
        'cv': distances_df['distance'].std() / distances_df['distance'].mean() if distances_df['distance'].mean() > 0 else 0
    }


def analyze_embedding_model(df, model_name, dim):
    """Analyze a single embedding model for cultural sensitivity."""
    embeddings = extract_embeddings(df)
    cultures = df['culture'].values
    story_types = df['story_type'].values
    
    # Overall inter-culture distances
    overall_distances = compute_inter_culture_distances(embeddings, cultures)
    
    # Per-story-type inter-culture distances
    by_story_type_distances = compute_inter_culture_distances_by_story_type(
        embeddings, cultures, story_types)
    
    # Sensitivity metrics
    overall_metrics = compute_sensitivity_metrics(overall_distances)
    by_st_metrics = compute_sensitivity_metrics(by_story_type_distances)
    
    return {
        'overall_distances': overall_distances,
        'by_story_type_distances': by_story_type_distances,
        'overall_metrics': overall_metrics,
        'by_story_type_metrics': by_st_metrics
    }


def compare_embedding_models(all_data):
    """Compare all embedding models on sensitivity metrics."""
    results = []
    all_distances = {}
    all_by_st_distances = {}
    
    for model_key, model_data in all_data.items():
        print(f"    Analyzing {model_data['name']}...")
        
        analysis = analyze_embedding_model(
            model_data['df'], 
            model_data['name'], 
            model_data['dim']
        )
        
        all_distances[model_key] = analysis['overall_distances']
        all_by_st_distances[model_key] = analysis['by_story_type_distances']
        
        results.append({
            'embedding_model': model_key,
            'short_name': model_data['name'],
            'dimensions': model_data['dim'],
            # Overall metrics
            'overall_mean_dist': analysis['overall_metrics']['mean_distance'],
            'overall_std_dist': analysis['overall_metrics']['std_distance'],
            'overall_variance': analysis['overall_metrics']['variance'],
            'overall_range': analysis['overall_metrics']['range'],
            'overall_cv': analysis['overall_metrics']['cv'],
            # Per-story-type metrics (measures within-type cultural separation)
            'by_st_mean_dist': analysis['by_story_type_metrics']['mean_distance'],
            'by_st_std_dist': analysis['by_story_type_metrics']['std_distance'],
            'by_st_variance': analysis['by_story_type_metrics']['variance'],
            'by_st_range': analysis['by_story_type_metrics']['range'],
            'by_st_cv': analysis['by_story_type_metrics']['cv']
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Add rankings
    comparison_df['variance_rank'] = comparison_df['overall_variance'].rank(ascending=False)
    comparison_df['range_rank'] = comparison_df['overall_range'].rank(ascending=False)
    comparison_df['cv_rank'] = comparison_df['overall_cv'].rank(ascending=False)
    
    # Composite sensitivity score (average of ranks)
    comparison_df['sensitivity_score'] = (
        comparison_df['variance_rank'] + 
        comparison_df['range_rank'] + 
        comparison_df['cv_rank']
    ) / 3
    comparison_df['sensitivity_rank'] = comparison_df['sensitivity_score'].rank()
    
    comparison_df = comparison_df.sort_values('sensitivity_rank')
    
    return comparison_df, all_distances, all_by_st_distances


def analyze_dimension_effect(comparison_df):
    """Analyze relationship between embedding dimensions and sensitivity."""
    # Correlation between dimensions and metrics
    dim_corr_variance = stats.pearsonr(
        comparison_df['dimensions'], 
        comparison_df['overall_variance']
    )
    dim_corr_range = stats.pearsonr(
        comparison_df['dimensions'], 
        comparison_df['overall_range']
    )
    dim_corr_mean = stats.pearsonr(
        comparison_df['dimensions'], 
        comparison_df['overall_mean_dist']
    )
    
    return {
        'dim_variance_r': dim_corr_variance[0],
        'dim_variance_p': dim_corr_variance[1],
        'dim_range_r': dim_corr_range[0],
        'dim_range_p': dim_corr_range[1],
        'dim_mean_r': dim_corr_mean[0],
        'dim_mean_p': dim_corr_mean[1]
    }


def create_visualizations(comparison_df, all_distances, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Bar chart: Sensitivity metrics by embedding model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sorted_df = comparison_df.sort_values('dimensions')
    
    # Variance
    axes[0].bar(sorted_df['short_name'], sorted_df['overall_variance'], color='steelblue')
    axes[0].set_title('Distance Variance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Embedding Model')
    axes[0].set_ylabel('Variance')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Range
    axes[1].bar(sorted_df['short_name'], sorted_df['overall_range'], color='darkorange')
    axes[1].set_title('Distance Range', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Embedding Model')
    axes[1].set_ylabel('Range')
    axes[1].tick_params(axis='x', rotation=45)
    
    # CV
    axes[2].bar(sorted_df['short_name'], sorted_df['overall_cv'], color='forestgreen')
    axes[2].set_title('Coefficient of Variation', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Embedding Model')
    axes[2].set_ylabel('CV')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('RQ8: Embedding Model Sensitivity Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq8_sensitivity_metrics.png', bbox_inches='tight')
    plt.close()
    
    # 2. Scatter: Dimensions vs Sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(comparison_df['dimensions'], comparison_df['overall_variance'], 
               s=150, c='steelblue', edgecolors='black', linewidth=2)
    
    for _, row in comparison_df.iterrows():
        ax.annotate(row['short_name'], 
                   (row['dimensions'], row['overall_variance']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add trend line
    z = np.polyfit(comparison_df['dimensions'], comparison_df['overall_variance'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(comparison_df['dimensions'].min(), comparison_df['dimensions'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    ax.set_xlabel('Embedding Dimensions', fontsize=11)
    ax.set_ylabel('Distance Variance (Sensitivity)', fontsize=11)
    ax.set_title('RQ8: Does Higher Dimensionality Capture More Nuance?', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq8_dimensions_vs_sensitivity.png', bbox_inches='tight')
    plt.close()
    
    # 3. Box plot: Distance distributions by embedding model
    fig, ax = plt.subplots(figsize=(12, 6))
    
    distance_data = []
    for model_key, distances in all_distances.items():
        model_name = EMBEDDING_MODELS[model_key]['name']
        for _, row in distances.iterrows():
            distance_data.append({
                'model': model_name,
                'distance': row['distance']
            })
    
    dist_df = pd.DataFrame(distance_data)
    
    # Order by dimensions
    dim_order = comparison_df.sort_values('dimensions')['short_name'].tolist()
    
    sns.boxplot(data=dist_df, x='model', y='distance', order=dim_order, palette='viridis', ax=ax)
    ax.set_title('RQ8: Inter-Culture Distance Distributions by Embedding Model', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Embedding Model (ordered by dimensions)', fontsize=11)
    ax.set_ylabel('Cosine Distance', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq8_distance_distributions.png', bbox_inches='tight')
    plt.close()
    
    # 4. Ranking visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranked_df = comparison_df.sort_values('sensitivity_rank')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_df)))
    
    bars = ax.barh(range(len(ranked_df)), ranked_df['sensitivity_score'], color=colors)
    ax.set_yticks(range(len(ranked_df)))
    ax.set_yticklabels([f"{row['short_name']} ({row['dimensions']}d)" 
                        for _, row in ranked_df.iterrows()])
    ax.set_xlabel('Sensitivity Score (lower = more sensitive)', fontsize=11)
    ax.set_title('RQ8: Embedding Model Ranking by Cultural Sensitivity', 
                 fontsize=14, fontweight='bold')
    
    # Add rank labels
    for i, (bar, score) in enumerate(zip(bars, ranked_df['sensitivity_score'])):
        ax.text(score + 0.05, i, f'#{int(ranked_df.iloc[i]["sensitivity_rank"])}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq8_model_ranking.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(comparison_df, dim_analysis):
    """Generate summary of key findings."""
    best_model = comparison_df.iloc[0]
    worst_model = comparison_df.iloc[-1]
    
    # Check dimension-sensitivity relationship
    dim_effect_significant = dim_analysis['dim_variance_p'] < 0.05
    dim_effect_direction = 'positive' if dim_analysis['dim_variance_r'] > 0 else 'negative'
    
    summary = {
        'research_question': 'RQ8: Embedding Model Sensitivity Comparison',
        'main_question': 'Does higher dimensionality capture more nuance?',
        'num_embedding_models': len(comparison_df),
        'dimension_range': f"{comparison_df['dimensions'].min()} - {comparison_df['dimensions'].max()}",
        
        # Best model
        'most_sensitive_model': best_model['short_name'],
        'most_sensitive_dims': int(best_model['dimensions']),
        'most_sensitive_variance': best_model['overall_variance'],
        'most_sensitive_range': best_model['overall_range'],
        
        # Worst model
        'least_sensitive_model': worst_model['short_name'],
        'least_sensitive_dims': int(worst_model['dimensions']),
        'least_sensitive_variance': worst_model['overall_variance'],
        
        # Dimension analysis
        'dim_variance_correlation': dim_analysis['dim_variance_r'],
        'dim_variance_pvalue': dim_analysis['dim_variance_p'],
        'dim_effect_significant': dim_effect_significant,
        'dim_effect_direction': dim_effect_direction,
        
        # Conclusion
        'conclusion': 'Higher dimensions = more sensitivity' if (dim_effect_significant and dim_effect_direction == 'positive') 
                      else 'Higher dimensions = less sensitivity' if (dim_effect_significant and dim_effect_direction == 'negative')
                      else 'No significant relationship between dimensions and sensitivity'
    }
    
    return summary


def main():
    """Main execution function for RQ8 analysis."""
    print("=" * 70)
    print("RQ8: Embedding Model Sensitivity Comparison")
    print("=" * 70)
    print("\nQuestion: Does higher dimensionality capture more nuance?")
    print("Method: Compare inter-culture distance variance across embedding models")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load all parquet files
    print("\n[1] Loading all embedding files...")
    all_data = load_all_parquet_files()
    print(f"    Loaded {len(all_data)} embedding models")
    
    # Compare embedding models
    print("\n[2] Analyzing embedding model sensitivity...")
    comparison_df, all_distances, all_by_st_distances = compare_embedding_models(all_data)
    
    # Export comparison results
    comparison_df.to_csv(output_dir / 'rq8_model_comparison.csv', index=False)
    print(f"    Exported: rq8_model_comparison.csv")
    
    print("\n    Embedding Model Sensitivity Ranking:")
    print("    " + "-" * 75)
    print(f"    {'Rank':<6} {'Model':<15} {'Dims':>6} {'Variance':>12} {'Range':>10} {'CV':>10}")
    print("    " + "-" * 75)
    for _, row in comparison_df.iterrows():
        print(f"    {int(row['sensitivity_rank']):<6} {row['short_name']:<15} {int(row['dimensions']):>6} "
              f"{row['overall_variance']:>12.6f} {row['overall_range']:>10.4f} {row['overall_cv']:>10.4f}")
    
    # Export detailed distances for each model
    print("\n[3] Exporting detailed distance data...")
    for model_key, distances in all_distances.items():
        model_name = EMBEDDING_MODELS[model_key]['name'].lower().replace('-', '_')
        distances.to_csv(output_dir / f'rq8_distances_{model_name}.csv', index=False)
    print(f"    Exported: rq8_distances_*.csv (5 files)")
    
    # Export by-story-type distances
    all_by_st_combined = []
    for model_key, distances in all_by_st_distances.items():
        distances = distances.copy()
        distances['embedding_model'] = EMBEDDING_MODELS[model_key]['name']
        all_by_st_combined.append(distances)
    
    combined_by_st = pd.concat(all_by_st_combined, ignore_index=True)
    combined_by_st.to_csv(output_dir / 'rq8_distances_by_story_type.csv', index=False)
    print(f"    Exported: rq8_distances_by_story_type.csv")
    
    # Analyze dimension effect
    print("\n[4] Analyzing dimension-sensitivity relationship...")
    dim_analysis = analyze_dimension_effect(comparison_df)
    
    dim_analysis_df = pd.DataFrame([dim_analysis])
    dim_analysis_df.to_csv(output_dir / 'rq8_dimension_analysis.csv', index=False)
    print(f"    Exported: rq8_dimension_analysis.csv")
    
    print(f"\n    Dimension vs Variance Correlation:")
    print(f"    r = {dim_analysis['dim_variance_r']:.4f}, p = {dim_analysis['dim_variance_p']:.4f}")
    sig = '*' if dim_analysis['dim_variance_p'] < 0.05 else ''
    print(f"    {'Significant' if sig else 'Not significant'} {sig}")
    
    # Create visualizations
    print("\n[5] Creating visualizations...")
    create_visualizations(comparison_df, all_distances, output_dir)
    
    # Generate summary
    print("\n[6] Generating summary...")
    summary = generate_summary(comparison_df, dim_analysis)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq8_summary.csv', index=False)
    print(f"    Exported: rq8_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ8 ANALYSIS COMPLETE")
    print("=" * 70)
    
    best = comparison_df.iloc[0]
    worst = comparison_df.iloc[-1]
    
    print(f"\nKey Findings:")
    print(f"\n  Embedding Model Ranking (by cultural sensitivity):")
    for _, row in comparison_df.iterrows():
        emoji = "🥇" if row['sensitivity_rank'] == 1 else "🥈" if row['sensitivity_rank'] == 2 else "🥉" if row['sensitivity_rank'] == 3 else "  "
        print(f"    {emoji} #{int(row['sensitivity_rank'])}: {row['short_name']} ({int(row['dimensions'])}d) - variance: {row['overall_variance']:.6f}")
    
    print(f"\n  Most Sensitive:  {best['short_name']} ({int(best['dimensions'])} dims)")
    print(f"  Least Sensitive: {worst['short_name']} ({int(worst['dimensions'])} dims)")
    
    print(f"\n  Dimension-Sensitivity Relationship:")
    print(f"    Correlation: r = {dim_analysis['dim_variance_r']:.4f}")
    print(f"    p-value:     {dim_analysis['dim_variance_p']:.4f}")
    
    if dim_analysis['dim_variance_p'] < 0.05:
        if dim_analysis['dim_variance_r'] > 0:
            print(f"\n  ✓ CONFIRMED: Higher dimensions DO capture more nuance")
        else:
            print(f"\n  ✗ OPPOSITE: Higher dimensions capture LESS nuance")
    else:
        print(f"\n  ⚠️  No significant relationship between dimensions and sensitivity")
        print(f"     Dimensionality alone does not determine cultural sensitivity")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq8_model_comparison.csv - Sensitivity metrics for all models")
    print("  2. rq8_distances_*.csv - Detailed pairwise distances (5 files)")
    print("  3. rq8_distances_by_story_type.csv - Distances per story type")
    print("  4. rq8_dimension_analysis.csv - Dimension correlation analysis")
    print("  5. rq8_summary.csv - Key findings summary")
    print("-" * 70)
    
    return comparison_df, dim_analysis


if __name__ == "__main__":
    comparison_df, dim_analysis = main()
