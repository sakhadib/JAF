"""
RQ10: Cross-Cultural Semantic Alignment in Story Types
======================================================
Question: Are "Origin Stories" (and other story types) universal or culturally distinct?
          Which story types show the most/least cultural variation?

Method:
- For each story_type, compute mean embedding (centroid) per culture
- Build distance matrix between culture centroids
- Compare across story types to identify which have most/least cultural variation
- High inter-cultural distance = culturally distinct narratives
- Low inter-cultural distance = universal/shared narratives

Data Source: Parquet (story_embedding)

Interpretation:
- High cultural variation: Cultures tell this story type very differently
- Low cultural variation: Universal themes, similar across cultures
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cosine, euclidean
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


def compute_culture_centroids(df, story_type):
    """
    Compute mean embedding (centroid) for each culture within a story type.
    """
    filtered = df[df['story_type'] == story_type]
    
    centroids = {}
    culture_counts = {}
    
    for culture in filtered['culture'].unique():
        culture_mask = filtered['culture'] == culture
        culture_embeddings = extract_embeddings(filtered[culture_mask])
        
        if len(culture_embeddings) > 0:
            centroids[culture] = culture_embeddings.mean(axis=0)
            culture_counts[culture] = len(culture_embeddings)
    
    return centroids, culture_counts


def compute_distance_matrix(centroids):
    """
    Compute pairwise distance matrix between culture centroids.
    Returns both Euclidean and Cosine distance matrices.
    """
    cultures = sorted(centroids.keys())
    n = len(cultures)
    
    euclidean_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(cultures):
        for j, c2 in enumerate(cultures):
            if i != j:
                euclidean_matrix[i, j] = euclidean(centroids[c1], centroids[c2])
                cosine_matrix[i, j] = cosine(centroids[c1], centroids[c2])  # Distance, not similarity
    
    euclidean_df = pd.DataFrame(euclidean_matrix, index=cultures, columns=cultures)
    cosine_df = pd.DataFrame(cosine_matrix, index=cultures, columns=cultures)
    
    return euclidean_df, cosine_df


def compute_cultural_variation(distance_matrix):
    """
    Compute summary statistics for cultural variation from distance matrix.
    Higher values = more culturally distinct narratives.
    """
    # Get upper triangle (excluding diagonal)
    upper_tri = distance_matrix.values[np.triu_indices(len(distance_matrix), k=1)]
    
    return {
        'mean_distance': np.mean(upper_tri),
        'std_distance': np.std(upper_tri),
        'min_distance': np.min(upper_tri),
        'max_distance': np.max(upper_tri),
        'median_distance': np.median(upper_tri),
        'range': np.max(upper_tri) - np.min(upper_tri)
    }


def analyze_all_story_types(df):
    """
    Analyze cultural variation for all story types.
    Returns comparison dataframe ranking story types by cultural distinctiveness.
    """
    story_types = df['story_type'].unique()
    
    results = []
    all_distance_matrices = {}
    all_centroids = {}
    
    for story_type in story_types:
        centroids, counts = compute_culture_centroids(df, story_type)
        
        if len(centroids) < 2:
            continue
        
        euclidean_matrix, cosine_matrix = compute_distance_matrix(centroids)
        
        # Store matrices
        all_distance_matrices[story_type] = {
            'euclidean': euclidean_matrix,
            'cosine': cosine_matrix
        }
        all_centroids[story_type] = centroids
        
        # Compute variation stats
        euclidean_stats = compute_cultural_variation(euclidean_matrix)
        cosine_stats = compute_cultural_variation(cosine_matrix)
        
        n_stories = df[df['story_type'] == story_type].shape[0]
        n_cultures = len(centroids)
        
        results.append({
            'story_type': story_type,
            'n_stories': n_stories,
            'n_cultures': n_cultures,
            'euclidean_mean': euclidean_stats['mean_distance'],
            'euclidean_std': euclidean_stats['std_distance'],
            'euclidean_min': euclidean_stats['min_distance'],
            'euclidean_max': euclidean_stats['max_distance'],
            'cosine_mean': cosine_stats['mean_distance'],
            'cosine_std': cosine_stats['std_distance'],
            'cosine_min': cosine_stats['min_distance'],
            'cosine_max': cosine_stats['max_distance'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Add rankings (1 = most distinct)
    results_df['euclidean_rank'] = results_df['euclidean_mean'].rank(ascending=False)
    results_df['cosine_rank'] = results_df['cosine_mean'].rank(ascending=False)
    
    # Combined score (average of both rankings)
    results_df['combined_rank'] = (results_df['euclidean_rank'] + results_df['cosine_rank']) / 2
    
    # Sort by combined distinctiveness
    results_df = results_df.sort_values('combined_rank')
    
    return results_df, all_distance_matrices, all_centroids


def identify_similar_cultures(df, story_type, top_n=5):
    """
    Identify the most similar culture pairs within a story type.
    """
    centroids, _ = compute_culture_centroids(df, story_type)
    _, cosine_matrix = compute_distance_matrix(centroids)
    
    pairs = []
    cultures = list(cosine_matrix.index)
    
    for i, c1 in enumerate(cultures):
        for j, c2 in enumerate(cultures):
            if i < j:
                pairs.append({
                    'story_type': story_type,
                    'culture_1': c1,
                    'culture_2': c2,
                    'cosine_distance': cosine_matrix.loc[c1, c2],
                    'similarity': 1 - cosine_matrix.loc[c1, c2]
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('cosine_distance')
    
    return pairs_df


def analyze_culture_pairs_all_types(df):
    """
    Find most similar and most different culture pairs across all story types.
    """
    all_pairs = []
    
    for story_type in df['story_type'].unique():
        pairs = identify_similar_cultures(df, story_type)
        all_pairs.append(pairs)
    
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)
    
    return all_pairs_df


def compute_universal_vs_distinct_scores(comparison_df):
    """
    Classify story types as "Universal" or "Culturally Distinct".
    """
    comparison_df = comparison_df.copy()
    
    # Normalize cosine mean to 0-1 scale for interpretability
    min_dist = comparison_df['cosine_mean'].min()
    max_dist = comparison_df['cosine_mean'].max()
    
    if max_dist > min_dist:
        comparison_df['distinctiveness_score'] = (comparison_df['cosine_mean'] - min_dist) / (max_dist - min_dist)
    else:
        comparison_df['distinctiveness_score'] = 0.5
    
    # Classify
    comparison_df['classification'] = comparison_df['distinctiveness_score'].apply(
        lambda x: 'Highly Distinct' if x > 0.66 else 'Moderate' if x > 0.33 else 'Universal'
    )
    
    return comparison_df


def perform_statistical_test(df, comparison_df):
    """
    Test if cultural variation differs significantly across story types.
    Uses ANOVA on pairwise distances.
    """
    # Collect all pairwise distances for each story type
    story_type_distances = {}
    
    for story_type in comparison_df['story_type']:
        centroids, _ = compute_culture_centroids(df, story_type)
        _, cosine_matrix = compute_distance_matrix(centroids)
        
        # Get upper triangle distances
        distances = cosine_matrix.values[np.triu_indices(len(cosine_matrix), k=1)]
        story_type_distances[story_type] = distances
    
    # ANOVA
    groups = list(story_type_distances.values())
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size
    all_distances = np.concatenate(groups)
    grand_mean = np.mean(all_distances)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((all_distances - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < 0.05
    }


def create_visualizations(df, comparison_df, all_distance_matrices, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Bar chart: Cultural variation by story type
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sorted_df = comparison_df.sort_values('cosine_mean', ascending=False)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_df)))
    
    bars = ax.bar(range(len(sorted_df)), sorted_df['cosine_mean'], 
                  yerr=sorted_df['cosine_std'], capsize=4, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df['story_type'], rotation=45, ha='right')
    ax.set_xlabel('Story Type', fontsize=11)
    ax.set_ylabel('Mean Inter-Cultural Distance (Cosine)', fontsize=11)
    ax.set_title('RQ10: Cultural Variation by Story Type\n(Higher = More Culturally Distinct)', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, sorted_df['cosine_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq10_cultural_variation_by_story_type.png', bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap for each story type (top 4 most and least distinct)
    most_distinct = sorted_df.head(2)['story_type'].tolist()
    least_distinct = sorted_df.tail(2)['story_type'].tolist()
    featured_types = most_distinct + least_distinct
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, story_type in enumerate(featured_types):
        if story_type in all_distance_matrices:
            cosine_matrix = all_distance_matrices[story_type]['cosine']
            
            sns.heatmap(cosine_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[idx], cbar_kws={'label': 'Cosine Distance'})
            
            label = "Most Distinct" if idx < 2 else "Most Universal"
            axes[idx].set_title(f'{story_type}\n({label})', fontsize=11, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].tick_params(axis='y', rotation=0)
    
    plt.suptitle('RQ10: Inter-Cultural Distance Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'rq10_distance_heatmaps.png', bbox_inches='tight')
    plt.close()
    
    # 3. Distinctiveness score visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    classified = compute_universal_vs_distinct_scores(comparison_df)
    classified = classified.sort_values('distinctiveness_score', ascending=True)
    
    colors_map = {'Highly Distinct': '#d62728', 'Moderate': '#ff7f0e', 'Universal': '#2ca02c'}
    bar_colors = [colors_map[c] for c in classified['classification']]
    
    bars = ax.barh(range(len(classified)), classified['distinctiveness_score'], 
                   color=bar_colors, edgecolor='black')
    
    ax.set_yticks(range(len(classified)))
    ax.set_yticklabels(classified['story_type'])
    ax.set_xlabel('Distinctiveness Score (0=Universal, 1=Highly Distinct)', fontsize=11)
    ax.set_title('RQ10: Story Type Classification by Cultural Distinctiveness', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Universal'),
        Patch(facecolor='#ff7f0e', label='Moderate'),
        Patch(facecolor='#d62728', label='Highly Distinct')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq10_distinctiveness_classification.png', bbox_inches='tight')
    plt.close()
    
    # 4. Culture similarity network (simplified as heatmap of average distances)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Aggregate distances across all story types
    cultures = df['culture'].unique()
    avg_distances = pd.DataFrame(0.0, index=cultures, columns=cultures)
    counts = pd.DataFrame(0, index=cultures, columns=cultures)
    
    for story_type, matrices in all_distance_matrices.items():
        cosine_matrix = matrices['cosine']
        for c1 in cosine_matrix.index:
            for c2 in cosine_matrix.columns:
                if c1 in avg_distances.index and c2 in avg_distances.columns:
                    avg_distances.loc[c1, c2] += cosine_matrix.loc[c1, c2]
                    counts.loc[c1, c2] += 1
    
    # Average
    avg_distances = avg_distances / counts.replace(0, 1)
    
    sns.heatmap(avg_distances, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Average Cosine Distance'})
    ax.set_title('RQ10: Average Inter-Cultural Distance Across All Story Types', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq10_culture_distance_aggregate.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(comparison_df, anova_result, all_pairs_df):
    """Generate summary of key findings."""
    classified = compute_universal_vs_distinct_scores(comparison_df)
    
    most_distinct = classified.iloc[classified['distinctiveness_score'].argmax()]
    most_universal = classified.iloc[classified['distinctiveness_score'].argmin()]
    
    # Most similar culture pair overall
    most_similar_pair = all_pairs_df.loc[all_pairs_df['cosine_distance'].idxmin()]
    
    # Most different culture pair overall
    most_different_pair = all_pairs_df.loc[all_pairs_df['cosine_distance'].idxmax()]
    
    summary = {
        'research_question': 'RQ10: Cross-Cultural Semantic Alignment in Story Types',
        'main_question': 'Are story types universal or culturally distinct?',
        'n_story_types': len(comparison_df),
        
        # Most distinct
        'most_distinct_type': most_distinct['story_type'],
        'most_distinct_score': most_distinct['distinctiveness_score'],
        'most_distinct_cosine_mean': most_distinct['cosine_mean'],
        
        # Most universal
        'most_universal_type': most_universal['story_type'],
        'most_universal_score': most_universal['distinctiveness_score'],
        'most_universal_cosine_mean': most_universal['cosine_mean'],
        
        # Culture pairs
        'most_similar_cultures': f"{most_similar_pair['culture_1']} & {most_similar_pair['culture_2']}",
        'most_similar_story_type': most_similar_pair['story_type'],
        'most_similar_distance': most_similar_pair['cosine_distance'],
        
        'most_different_cultures': f"{most_different_pair['culture_1']} & {most_different_pair['culture_2']}",
        'most_different_story_type': most_different_pair['story_type'],
        'most_different_distance': most_different_pair['cosine_distance'],
        
        # ANOVA
        'anova_f_statistic': anova_result['f_statistic'],
        'anova_p_value': anova_result['p_value'],
        'anova_eta_squared': anova_result['eta_squared'],
        'anova_significant': anova_result['significant']
    }
    
    return summary


def main():
    """Main execution function for RQ10 analysis."""
    print("=" * 70)
    print("RQ10: Cross-Cultural Semantic Alignment in Story Types")
    print("=" * 70)
    print("\nQuestion: Are story types universal or culturally distinct?")
    print("Method: Inter-cultural distance analysis across story types")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading embedding data...")
    df = load_parquet_data('openai_text-embedding-3-large')
    print(f"    Loaded {len(df)} stories")
    print(f"    Story types: {df['story_type'].nunique()}")
    print(f"    Cultures: {df['culture'].nunique()}")
    
    # Analyze all story types
    print("\n[2] Computing cultural variation for each story type...")
    comparison_df, all_distance_matrices, all_centroids = analyze_all_story_types(df)
    comparison_df.to_csv(output_dir / 'rq10_story_type_comparison.csv', index=False)
    print(f"    Exported: rq10_story_type_comparison.csv")
    
    print("\n    Story Type Cultural Variation Ranking:")
    print("    " + "-" * 70)
    print(f"    {'Rank':<6} {'Story Type':<35} {'Cosine Mean':>12} {'Std':>10}")
    print("    " + "-" * 70)
    for idx, row in comparison_df.iterrows():
        print(f"    {int(row['combined_rank']):<6} {row['story_type']:<35} "
              f"{row['cosine_mean']:>12.4f} {row['cosine_std']:>10.4f}")
    
    # Export distance matrices for each story type
    print("\n[3] Exporting distance matrices...")
    for story_type, matrices in all_distance_matrices.items():
        safe_name = story_type.lower().replace(' ', '_').replace('/', '_')
        matrices['cosine'].to_csv(output_dir / f'rq10_distance_matrix_{safe_name}.csv')
    print(f"    Exported {len(all_distance_matrices)} distance matrices")
    
    # Analyze culture pairs
    print("\n[4] Analyzing culture pairs across all story types...")
    all_pairs_df = analyze_culture_pairs_all_types(df)
    all_pairs_df.to_csv(output_dir / 'rq10_culture_pairs.csv', index=False)
    print(f"    Exported: rq10_culture_pairs.csv")
    print(f"    Total pairs analyzed: {len(all_pairs_df)}")
    
    # Most similar and different
    most_similar = all_pairs_df.nsmallest(5, 'cosine_distance')
    most_different = all_pairs_df.nlargest(5, 'cosine_distance')
    
    print("\n    Top 5 Most Similar Culture Pairs:")
    for _, row in most_similar.iterrows():
        print(f"      {row['culture_1']} & {row['culture_2']} in {row['story_type']}: {row['cosine_distance']:.4f}")
    
    print("\n    Top 5 Most Different Culture Pairs:")
    for _, row in most_different.iterrows():
        print(f"      {row['culture_1']} & {row['culture_2']} in {row['story_type']}: {row['cosine_distance']:.4f}")
    
    # Classification
    print("\n[5] Classifying story types...")
    classified_df = compute_universal_vs_distinct_scores(comparison_df)
    classified_df.to_csv(output_dir / 'rq10_story_type_classification.csv', index=False)
    print(f"    Exported: rq10_story_type_classification.csv")
    
    # Count by classification
    class_counts = classified_df['classification'].value_counts()
    print("\n    Classification Distribution:")
    for cls, count in class_counts.items():
        print(f"      {cls}: {count} story types")
    
    # ANOVA test
    print("\n[6] Performing ANOVA test...")
    anova_result = perform_statistical_test(df, comparison_df)
    
    anova_df = pd.DataFrame([anova_result])
    anova_df.to_csv(output_dir / 'rq10_anova.csv', index=False)
    print(f"    Exported: rq10_anova.csv")
    
    print(f"\n    ANOVA Results:")
    print(f"    F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"    p-value:     {anova_result['p_value']:.6e}")
    print(f"    η² (effect): {anova_result['eta_squared']:.4f}")
    sig = '***' if anova_result['p_value'] < 0.001 else '**' if anova_result['p_value'] < 0.01 else '*' if anova_result['p_value'] < 0.05 else ''
    print(f"    Significant: {'Yes' if anova_result['significant'] else 'No'} {sig}")
    
    # Create visualizations
    print("\n[7] Creating visualizations...")
    create_visualizations(df, comparison_df, all_distance_matrices, output_dir)
    
    # Generate summary
    print("\n[8] Generating summary...")
    summary = generate_summary(comparison_df, anova_result, all_pairs_df)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq10_summary.csv', index=False)
    print(f"    Exported: rq10_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ10 ANALYSIS COMPLETE")
    print("=" * 70)
    
    classified = compute_universal_vs_distinct_scores(comparison_df)
    most_distinct = classified.loc[classified['distinctiveness_score'].idxmax()]
    most_universal = classified.loc[classified['distinctiveness_score'].idxmin()]
    
    print(f"\nKey Findings:")
    print(f"\n  Story Type Cultural Distinctiveness Ranking:")
    for idx, row in classified.sort_values('distinctiveness_score', ascending=False).iterrows():
        emoji = "🔴" if row['classification'] == 'Highly Distinct' else "🟡" if row['classification'] == 'Moderate' else "🟢"
        print(f"    {emoji} {row['story_type']}: {row['distinctiveness_score']:.3f} ({row['classification']})")
    
    print(f"\n  Most Culturally Distinct: {most_distinct['story_type']}")
    print(f"    → Cultures tell this story type very differently")
    
    print(f"\n  Most Universal: {most_universal['story_type']}")
    print(f"    → Shared narrative themes across cultures")
    
    if anova_result['significant']:
        print(f"\n  ✓ SIGNIFICANT difference in cultural variation across story types (p < 0.05)")
        print(f"    Some story types are more culturally influenced than others.")
    else:
        print(f"\n  ✗ No significant difference across story types (p ≥ 0.05)")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq10_story_type_comparison.csv - Main comparison metrics")
    print("  2. rq10_culture_pairs.csv - All culture pair distances")
    print("  3. rq10_story_type_classification.csv - Universal/Distinct classification")
    print("  4. rq10_anova.csv - ANOVA test results")
    print("  5. rq10_summary.csv - Key findings summary")
    print(f"  6-{5+len(all_distance_matrices)}. rq10_distance_matrix_*.csv - Per story type matrices")
    print("-" * 70)
    
    return df, comparison_df, anova_result


if __name__ == "__main__":
    df, comparison_df, anova_result = main()
