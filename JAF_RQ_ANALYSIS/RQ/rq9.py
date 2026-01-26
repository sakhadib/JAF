"""
RQ9: Detection of Model Collapse / Repetition
==============================================
Question: Do certain LLM generation models produce semantically repetitive outputs?
          Which models show signs of "model collapse" or generic story generation?

Method:
- Group by `model` to get all story_embeddings per generation model
- Intra-model similarity: Average pairwise cosine similarity within each model
- High similarity = repetitive/generic outputs (model collapse)
- Identify which model has tightest cluster (most repetitive)

Data Source: Parquet (story_embedding)

Interpretation:
- High intra-model similarity → Model produces similar/repetitive stories
- Low intra-model similarity → Model produces diverse/varied stories
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


def load_parquet_data(embedding_model='openai_text-embedding-3-large'):
    """Load parquet data with story embeddings."""
    parquet_dir = Path(__file__).parent.parent / "data" / "parquet"
    parquet_file = parquet_dir / f"{embedding_model}.parquet"
    
    if not parquet_file.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/parquet/")
        parquet_file = parquet_files[0]
        print(f"    Using fallback: {parquet_file.name}")
    
    df = pd.read_parquet(parquet_file)
    return df


def extract_embeddings(df):
    """Extract story embeddings as numpy array."""
    return np.array(df['story_embedding'].tolist())


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def compute_intra_model_similarity(df):
    """
    Compute average pairwise cosine similarity within each generation model.
    High similarity = repetitive/generic outputs.
    """
    results = []
    
    for model in df['model'].unique():
        model_mask = df['model'] == model
        model_df = df[model_mask]
        model_embeddings = extract_embeddings(model_df)
        
        n_stories = len(model_embeddings)
        
        # Compute pairwise cosine similarities
        # Using pdist for efficiency
        pairwise_distances = pdist(model_embeddings, metric='cosine')
        pairwise_similarities = 1 - pairwise_distances  # Convert distance to similarity
        
        # Statistics
        mean_sim = np.mean(pairwise_similarities)
        std_sim = np.std(pairwise_similarities)
        min_sim = np.min(pairwise_similarities)
        max_sim = np.max(pairwise_similarities)
        median_sim = np.median(pairwise_similarities)
        
        # Number of pairs
        n_pairs = len(pairwise_similarities)
        
        results.append({
            'model': model,
            'n_stories': n_stories,
            'n_pairs': n_pairs,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'median_similarity': median_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'similarity_range': max_sim - min_sim
        })
    
    results_df = pd.DataFrame(results)
    
    # Add repetitiveness score (normalized mean similarity)
    # Higher = more repetitive
    results_df['repetitiveness_score'] = results_df['mean_similarity']
    
    # Rank by repetitiveness (1 = most repetitive)
    results_df['repetitiveness_rank'] = results_df['repetitiveness_score'].rank(ascending=False)
    
    # Sort by repetitiveness
    results_df = results_df.sort_values('repetitiveness_score', ascending=False)
    
    return results_df


def compute_intra_model_similarity_by_culture(df):
    """
    Compute intra-model similarity separately for each culture.
    Helps identify if repetitiveness varies by culture.
    """
    results = []
    
    for model in df['model'].unique():
        for culture in df['culture'].unique():
            mask = (df['model'] == model) & (df['culture'] == culture)
            subset_df = df[mask]
            
            if len(subset_df) < 2:
                continue
            
            embeddings = extract_embeddings(subset_df)
            
            pairwise_distances = pdist(embeddings, metric='cosine')
            pairwise_similarities = 1 - pairwise_distances
            
            if len(pairwise_similarities) > 0:
                results.append({
                    'model': model,
                    'culture': culture,
                    'n_stories': len(subset_df),
                    'n_pairs': len(pairwise_similarities),
                    'mean_similarity': np.mean(pairwise_similarities),
                    'std_similarity': np.std(pairwise_similarities)
                })
    
    return pd.DataFrame(results)


def compute_intra_model_similarity_by_story_type(df):
    """
    Compute intra-model similarity separately for each story type.
    Helps identify if certain story types lead to more repetitive outputs.
    """
    results = []
    
    for model in df['model'].unique():
        for story_type in df['story_type'].unique():
            mask = (df['model'] == model) & (df['story_type'] == story_type)
            subset_df = df[mask]
            
            if len(subset_df) < 2:
                continue
            
            embeddings = extract_embeddings(subset_df)
            
            pairwise_distances = pdist(embeddings, metric='cosine')
            pairwise_similarities = 1 - pairwise_distances
            
            if len(pairwise_similarities) > 0:
                results.append({
                    'model': model,
                    'story_type': story_type,
                    'n_stories': len(subset_df),
                    'n_pairs': len(pairwise_similarities),
                    'mean_similarity': np.mean(pairwise_similarities),
                    'std_similarity': np.std(pairwise_similarities)
                })
    
    return pd.DataFrame(results)


def perform_anova_repetitiveness(df):
    """Test if repetitiveness differs significantly across models."""
    # We need to compare distributions, so we'll sample pairwise similarities
    model_similarities = {}
    
    for model in df['model'].unique():
        model_mask = df['model'] == model
        model_embeddings = extract_embeddings(df[model_mask])
        
        pairwise_distances = pdist(model_embeddings, metric='cosine')
        pairwise_similarities = 1 - pairwise_distances
        
        model_similarities[model] = pairwise_similarities
    
    # ANOVA on similarity distributions
    groups = list(model_similarities.values())
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    all_sims = np.concatenate(groups)
    grand_mean = np.mean(all_sims)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((all_sims - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < 0.05
    }


def detect_outliers(model_stats):
    """
    Detect outlier models (significantly more or less repetitive).
    Uses IQR method.
    """
    mean_sims = model_stats['mean_similarity']
    
    q1 = mean_sims.quantile(0.25)
    q3 = mean_sims.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = model_stats[
        (model_stats['mean_similarity'] < lower_bound) | 
        (model_stats['mean_similarity'] > upper_bound)
    ].copy()
    
    outliers['outlier_type'] = outliers['mean_similarity'].apply(
        lambda x: 'High (Repetitive)' if x > upper_bound else 'Low (Diverse)'
    )
    
    return outliers, lower_bound, upper_bound


def compute_diversity_metrics(df):
    """
    Compute additional diversity metrics for each model.
    """
    results = []
    
    for model in df['model'].unique():
        model_mask = df['model'] == model
        model_embeddings = extract_embeddings(df[model_mask])
        
        # Centroid
        centroid = model_embeddings.mean(axis=0)
        
        # Distance from centroid
        distances_from_centroid = np.linalg.norm(model_embeddings - centroid, axis=1)
        
        # Diversity = spread from centroid
        mean_spread = np.mean(distances_from_centroid)
        std_spread = np.std(distances_from_centroid)
        
        # Embedding variance (average variance across dimensions)
        embedding_variance = np.var(model_embeddings, axis=0).mean()
        
        results.append({
            'model': model,
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'embedding_variance': embedding_variance,
            'diversity_score': mean_spread  # Higher spread = more diverse
        })
    
    results_df = pd.DataFrame(results)
    results_df['diversity_rank'] = results_df['diversity_score'].rank(ascending=False)
    results_df = results_df.sort_values('diversity_score', ascending=False)
    
    return results_df


def create_visualizations(df, model_stats, diversity_stats, by_culture, by_story_type, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Bar chart: Repetitiveness by model
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_stats = model_stats.sort_values('mean_similarity', ascending=False)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_stats)))
    
    bars = ax.bar(range(len(sorted_stats)), sorted_stats['mean_similarity'], 
                  yerr=sorted_stats['std_similarity'], capsize=5, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(sorted_stats)))
    ax.set_xticklabels(sorted_stats['model'], rotation=45, ha='right')
    ax.set_xlabel('Generation Model', fontsize=11)
    ax.set_ylabel('Mean Intra-Model Similarity', fontsize=11)
    ax.set_title('RQ9: Model Repetitiveness (Higher = More Repetitive)', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, sorted_stats['mean_similarity']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq9_repetitiveness_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 2. Box plot: Similarity distributions by model
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect pairwise similarities for box plot
    sim_data = []
    for model in df['model'].unique():
        model_mask = df['model'] == model
        model_embeddings = extract_embeddings(df[model_mask])
        pairwise_distances = pdist(model_embeddings, metric='cosine')
        pairwise_similarities = 1 - pairwise_distances
        
        for sim in pairwise_similarities:
            sim_data.append({'model': model, 'similarity': sim})
    
    sim_df = pd.DataFrame(sim_data)
    
    order = sorted_stats['model'].tolist()
    sns.boxplot(data=sim_df, x='model', y='similarity', order=order, palette='viridis', ax=ax)
    
    ax.set_xlabel('Generation Model', fontsize=11)
    ax.set_ylabel('Pairwise Cosine Similarity', fontsize=11)
    ax.set_title('RQ9: Intra-Model Similarity Distributions', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq9_similarity_distributions.png', bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: Model × Culture repetitiveness
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot = by_culture.pivot_table(
        values='mean_similarity',
        index='model',
        columns='culture',
        aggfunc='mean'
    )
    
    # Sort by overall mean
    pivot['Overall'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Overall', ascending=False)
    pivot = pivot.drop(columns=['Overall'])
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Mean Similarity'})
    ax.set_title('RQ9: Repetitiveness by Model × Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq9_heatmap_model_culture.png', bbox_inches='tight')
    plt.close()
    
    # 4. Diversity vs Repetitiveness scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    merged = model_stats.merge(diversity_stats[['model', 'diversity_score']], on='model')
    
    ax.scatter(merged['diversity_score'], merged['mean_similarity'], 
               s=150, c='steelblue', edgecolors='black', linewidth=2)
    
    for _, row in merged.iterrows():
        ax.annotate(row['model'], (row['diversity_score'], row['mean_similarity']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Diversity Score (Spread from Centroid)', fontsize=11)
    ax.set_ylabel('Repetitiveness (Mean Intra-Model Similarity)', fontsize=11)
    ax.set_title('RQ9: Diversity vs Repetitiveness', fontsize=14, fontweight='bold')
    
    # Add quadrant labels
    x_mid = (merged['diversity_score'].max() + merged['diversity_score'].min()) / 2
    y_mid = (merged['mean_similarity'].max() + merged['mean_similarity'].min()) / 2
    
    ax.axhline(y=y_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=x_mid, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq9_diversity_vs_repetitiveness.png', bbox_inches='tight')
    plt.close()
    
    # 5. Story type effect on repetitiveness
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot_st = by_story_type.pivot_table(
        values='mean_similarity',
        index='model',
        columns='story_type',
        aggfunc='mean'
    )
    
    pivot_st['Overall'] = pivot_st.mean(axis=1)
    pivot_st = pivot_st.sort_values('Overall', ascending=False)
    pivot_st = pivot_st.drop(columns=['Overall'])
    
    sns.heatmap(pivot_st, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Mean Similarity'})
    ax.set_title('RQ9: Repetitiveness by Model × Story Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Story Type', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq9_heatmap_model_storytype.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(model_stats, diversity_stats, anova_result, outliers):
    """Generate summary of key findings."""
    most_repetitive = model_stats.iloc[0]
    least_repetitive = model_stats.iloc[-1]
    
    summary = {
        'research_question': 'RQ9: Detection of Model Collapse / Repetition',
        'main_question': 'Do certain models produce semantically repetitive outputs?',
        'total_models': len(model_stats),
        
        # Most repetitive model
        'most_repetitive_model': most_repetitive['model'],
        'most_repetitive_similarity': most_repetitive['mean_similarity'],
        'most_repetitive_std': most_repetitive['std_similarity'],
        
        # Least repetitive model
        'least_repetitive_model': least_repetitive['model'],
        'least_repetitive_similarity': least_repetitive['mean_similarity'],
        'least_repetitive_std': least_repetitive['std_similarity'],
        
        # Gap
        'repetitiveness_gap': most_repetitive['mean_similarity'] - least_repetitive['mean_similarity'],
        
        # ANOVA
        'anova_f_statistic': anova_result['f_statistic'],
        'anova_p_value': anova_result['p_value'],
        'anova_eta_squared': anova_result['eta_squared'],
        'anova_significant': anova_result['significant'],
        
        # Outliers
        'num_outliers': len(outliers),
        'outlier_models': outliers['model'].tolist() if len(outliers) > 0 else []
    }
    
    return summary


def main():
    """Main execution function for RQ9 analysis."""
    print("=" * 70)
    print("RQ9: Detection of Model Collapse / Repetition")
    print("=" * 70)
    print("\nQuestion: Do certain models produce semantically repetitive outputs?")
    print("Method: Intra-model pairwise cosine similarity analysis")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading embedding data...")
    df = load_parquet_data('openai_text-embedding-3-large')
    print(f"    Loaded {len(df)} stories")
    print(f"    Models: {df['model'].nunique()}")
    
    # Compute intra-model similarity
    print("\n[2] Computing intra-model similarity (repetitiveness)...")
    model_stats = compute_intra_model_similarity(df)
    model_stats.to_csv(output_dir / 'rq9_model_repetitiveness.csv', index=False)
    print(f"    Exported: rq9_model_repetitiveness.csv")
    
    print("\n    Model Repetitiveness Ranking (Higher = More Repetitive):")
    print("    " + "-" * 70)
    print(f"    {'Rank':<6} {'Model':<40} {'Mean Sim':>12} {'Std':>10}")
    print("    " + "-" * 70)
    for _, row in model_stats.iterrows():
        print(f"    {int(row['repetitiveness_rank']):<6} {row['model']:<40} "
              f"{row['mean_similarity']:>12.4f} {row['std_similarity']:>10.4f}")
    
    # Compute by culture
    print("\n[3] Computing repetitiveness by culture...")
    by_culture = compute_intra_model_similarity_by_culture(df)
    by_culture.to_csv(output_dir / 'rq9_repetitiveness_by_culture.csv', index=False)
    print(f"    Exported: rq9_repetitiveness_by_culture.csv")
    
    # Compute by story type
    print("\n[4] Computing repetitiveness by story type...")
    by_story_type = compute_intra_model_similarity_by_story_type(df)
    by_story_type.to_csv(output_dir / 'rq9_repetitiveness_by_story_type.csv', index=False)
    print(f"    Exported: rq9_repetitiveness_by_story_type.csv")
    
    # Compute diversity metrics
    print("\n[5] Computing diversity metrics...")
    diversity_stats = compute_diversity_metrics(df)
    diversity_stats.to_csv(output_dir / 'rq9_diversity_metrics.csv', index=False)
    print(f"    Exported: rq9_diversity_metrics.csv")
    
    # ANOVA test
    print("\n[6] Performing ANOVA test...")
    anova_result = perform_anova_repetitiveness(df)
    
    anova_df = pd.DataFrame([anova_result])
    anova_df.to_csv(output_dir / 'rq9_anova.csv', index=False)
    print(f"    Exported: rq9_anova.csv")
    
    print(f"\n    ANOVA Results:")
    print(f"    F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"    p-value:     {anova_result['p_value']:.6e}")
    print(f"    η² (effect): {anova_result['eta_squared']:.4f}")
    sig = '***' if anova_result['p_value'] < 0.001 else '**' if anova_result['p_value'] < 0.01 else '*' if anova_result['p_value'] < 0.05 else ''
    print(f"    Significant: {'Yes' if anova_result['significant'] else 'No'} {sig}")
    
    # Outlier detection
    print("\n[7] Detecting outlier models...")
    outliers, lower_bound, upper_bound = detect_outliers(model_stats)
    
    if len(outliers) > 0:
        outliers.to_csv(output_dir / 'rq9_outliers.csv', index=False)
        print(f"    Exported: rq9_outliers.csv")
        print(f"    Outliers detected: {len(outliers)}")
        for _, row in outliers.iterrows():
            print(f"      - {row['model']}: {row['outlier_type']}")
    else:
        print("    No significant outliers detected")
    
    # Create visualizations
    print("\n[8] Creating visualizations...")
    create_visualizations(df, model_stats, diversity_stats, by_culture, by_story_type, output_dir)
    
    # Generate summary
    print("\n[9] Generating summary...")
    summary = generate_summary(model_stats, diversity_stats, anova_result, outliers)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq9_summary.csv', index=False)
    print(f"    Exported: rq9_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ9 ANALYSIS COMPLETE")
    print("=" * 70)
    
    most_rep = model_stats.iloc[0]
    least_rep = model_stats.iloc[-1]
    
    print(f"\nKey Findings:")
    print(f"\n  Model Repetitiveness Ranking:")
    for _, row in model_stats.iterrows():
        emoji = "🔴" if row['repetitiveness_rank'] == 1 else "🟡" if row['repetitiveness_rank'] == 2 else "🟢" if row['repetitiveness_rank'] == len(model_stats) else "⚪"
        print(f"    {emoji} #{int(row['repetitiveness_rank'])}: {row['model']} (sim: {row['mean_similarity']:.4f})")
    
    print(f"\n  Most Repetitive:  {most_rep['model']} ({most_rep['mean_similarity']:.4f})")
    print(f"  Least Repetitive: {least_rep['model']} ({least_rep['mean_similarity']:.4f})")
    print(f"  Gap:              {most_rep['mean_similarity'] - least_rep['mean_similarity']:.4f}")
    
    if anova_result['significant']:
        print(f"\n  ✓ SIGNIFICANT difference in repetitiveness across models (p < 0.05)")
        print(f"    Some models produce more repetitive/generic outputs than others.")
    else:
        print(f"\n  ✗ No significant difference in repetitiveness (p ≥ 0.05)")
    
    # Interpretation
    if most_rep['mean_similarity'] > 0.5:
        print(f"\n  ⚠️  WARNING: {most_rep['model']} shows potential model collapse!")
        print(f"     High similarity ({most_rep['mean_similarity']:.4f}) suggests repetitive outputs.")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq9_model_repetitiveness.csv - Main repetitiveness metrics")
    print("  2. rq9_repetitiveness_by_culture.csv - By culture breakdown")
    print("  3. rq9_repetitiveness_by_story_type.csv - By story type breakdown")
    print("  4. rq9_diversity_metrics.csv - Diversity/spread metrics")
    print("  5. rq9_anova.csv - ANOVA test results")
    print("  6. rq9_outliers.csv - Outlier models (if any)")
    print("  7. rq9_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, model_stats, anova_result


if __name__ == "__main__":
    df, model_stats, anova_result = main()
