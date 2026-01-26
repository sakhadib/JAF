"""
RQ14: The "Safety Filter" Effect
================================
Question: Do generic/safe outputs receive lower cultural scores?
          Does playing it safe semantically lead to lower cultural authenticity?

Method:
- Compute "genericity": Similarity of each story to the overall embedding centroid
- Identify generic stories: Top 10% most similar to centroid (least unique)
- Compare quality scores: Generic vs. non-generic stories
- Per-model analysis: Which model produces most generic outputs?

Data Source: Parquet (story_embedding) + CSV (evaluation scores)

Interpretation:
- High genericity = story is very similar to average/typical story
- If generic stories have lower cultural scores → "safety filter" penalty exists
- Some models may produce more generic outputs due to safety mechanisms
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
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


# Target metrics (actual column names from CSV)
TARGET_METRICS = [
    'Cultural Accuracy & Authenticity',
    'Contextual & Temporal Appropriateness',
    'Narrative & Symbolic Coherence',
    'Linguistic & Expressive Appropriateness (Bangla)'
]

# Shorter display names
METRIC_SHORT_NAMES = {
    'Cultural Accuracy & Authenticity': 'cultural_accuracy',
    'Contextual & Temporal Appropriateness': 'contextual_appropriateness',
    'Narrative & Symbolic Coherence': 'narrative_coherence',
    'Linguistic & Expressive Appropriateness (Bangla)': 'linguistic_fluency'
}


def load_data():
    """Load and merge embedding data with evaluation scores."""
    data_dir = Path(__file__).parent.parent / "data"
    parquet_dir = data_dir / "parquet"
    
    # Load CSV with scores
    csv_path = data_dir / "evaluated_stories.csv"
    csv_df = pd.read_csv(csv_path)
    
    # Load parquet with embeddings
    parquet_file = parquet_dir / "openai_text-embedding-3-large.parquet"
    if not parquet_file.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            parquet_file = parquet_files[0]
    
    parquet_df = pd.read_parquet(parquet_file)
    
    # Merge on common columns
    merged = csv_df.merge(
        parquet_df[['model', 'culture', 'story', 'story_type', 'region', 'story_embedding']],
        on=['model', 'culture', 'story'],
        how='inner'
    )
    
    return merged


def compute_overall_centroid(df):
    """Compute the overall centroid of all story embeddings."""
    embeddings = np.array(df['story_embedding'].tolist())
    centroid = embeddings.mean(axis=0)
    return centroid


def compute_genericity(df, centroid):
    """
    Compute genericity score for each story.
    Genericity = similarity to overall centroid (higher = more generic)
    """
    df = df.copy()
    
    similarities = []
    distances = []
    
    for idx, row in df.iterrows():
        emb = np.array(row['story_embedding'])
        
        # Cosine similarity to centroid (higher = more generic)
        similarity = 1 - cosine(emb, centroid)
        similarities.append(similarity)
        
        # Euclidean distance (lower = more generic)
        dist = euclidean(emb, centroid)
        distances.append(dist)
    
    df['centroid_similarity'] = similarities
    df['centroid_distance'] = distances
    
    # Normalize genericity to 0-1 scale
    min_sim, max_sim = min(similarities), max(similarities)
    df['genericity_score'] = (df['centroid_similarity'] - min_sim) / (max_sim - min_sim)
    
    # Compute average score
    df['average_score'] = df[TARGET_METRICS].mean(axis=1)
    
    return df


def identify_generic_stories(df, percentile=10):
    """
    Identify generic stories as top X% most similar to centroid.
    """
    df = df.copy()
    
    threshold = df['genericity_score'].quantile(1 - percentile/100)
    df['is_generic'] = df['genericity_score'] >= threshold
    
    return df, threshold


def compare_generic_vs_nongeneric(df):
    """Compare quality scores between generic and non-generic stories."""
    results = []
    
    generic = df[df['is_generic']]
    nongeneric = df[~df['is_generic']]
    
    for metric in TARGET_METRICS + ['average_score']:
        short_name = METRIC_SHORT_NAMES.get(metric, 'average_score')
        
        generic_mean = generic[metric].mean()
        generic_std = generic[metric].std()
        nongeneric_mean = nongeneric[metric].mean()
        nongeneric_std = nongeneric[metric].std()
        
        # t-test
        t_stat, p_value = stats.ttest_ind(generic[metric], nongeneric[metric])
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(generic)-1)*generic_std**2 + (len(nongeneric)-1)*nongeneric_std**2) 
            / (len(generic) + len(nongeneric) - 2)
        )
        cohens_d = (nongeneric_mean - generic_mean) / pooled_std if pooled_std > 0 else 0
        
        # Genericity penalty (positive = generic stories score lower)
        penalty = nongeneric_mean - generic_mean
        
        results.append({
            'metric': metric,
            'metric_short': short_name,
            'generic_count': len(generic),
            'nongeneric_count': len(nongeneric),
            'generic_mean': generic_mean,
            'generic_std': generic_std,
            'nongeneric_mean': nongeneric_mean,
            'nongeneric_std': nongeneric_std,
            'genericity_penalty': penalty,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)


def correlate_genericity_with_scores(df):
    """Compute correlation between genericity and quality scores."""
    results = []
    
    for metric in TARGET_METRICS + ['average_score']:
        short_name = METRIC_SHORT_NAMES.get(metric, 'average_score')
        
        # Pearson correlation
        r, p = stats.pearsonr(df['genericity_score'], df[metric])
        
        # Spearman correlation
        rho, p_spearman = stats.spearmanr(df['genericity_score'], df[metric])
        
        results.append({
            'metric': metric,
            'metric_short': short_name,
            'pearson_r': r,
            'pearson_p_value': p,
            'pearson_significant': p < 0.05,
            'spearman_rho': rho,
            'spearman_p_value': p_spearman,
            'spearman_significant': p_spearman < 0.05
        })
    
    return pd.DataFrame(results)


def analyze_genericity_by_model(df):
    """Analyze which model produces most generic outputs."""
    results = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        mean_genericity = model_data['genericity_score'].mean()
        std_genericity = model_data['genericity_score'].std()
        pct_generic = model_data['is_generic'].mean() * 100
        
        # Mean quality scores
        mean_cultural = model_data['Cultural Accuracy & Authenticity'].mean()
        mean_avg_score = model_data['average_score'].mean()
        
        # Correlation between genericity and cultural score for this model
        r, p = stats.pearsonr(model_data['genericity_score'], 
                             model_data['Cultural Accuracy & Authenticity'])
        
        results.append({
            'model': model,
            'n_stories': len(model_data),
            'mean_genericity': mean_genericity,
            'std_genericity': std_genericity,
            'pct_generic': pct_generic,
            'mean_cultural_accuracy': mean_cultural,
            'mean_average_score': mean_avg_score,
            'genericity_cultural_correlation': r,
            'correlation_p_value': p,
            'correlation_significant': p < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df['genericity_rank'] = results_df['mean_genericity'].rank(ascending=False)
    results_df = results_df.sort_values('mean_genericity', ascending=False)
    
    return results_df


def analyze_genericity_by_culture(df):
    """Analyze genericity patterns by culture."""
    results = []
    
    for culture in df['culture'].unique():
        culture_data = df[df['culture'] == culture]
        
        mean_genericity = culture_data['genericity_score'].mean()
        std_genericity = culture_data['genericity_score'].std()
        pct_generic = culture_data['is_generic'].mean() * 100
        
        mean_cultural = culture_data['Cultural Accuracy & Authenticity'].mean()
        
        results.append({
            'culture': culture,
            'n_stories': len(culture_data),
            'mean_genericity': mean_genericity,
            'std_genericity': std_genericity,
            'pct_generic': pct_generic,
            'mean_cultural_accuracy': mean_cultural
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_genericity', ascending=False)
    
    return results_df


def analyze_genericity_by_story_type(df):
    """Analyze genericity patterns by story type."""
    results = []
    
    for story_type in df['story_type'].unique():
        st_data = df[df['story_type'] == story_type]
        
        mean_genericity = st_data['genericity_score'].mean()
        std_genericity = st_data['genericity_score'].std()
        pct_generic = st_data['is_generic'].mean() * 100
        
        mean_cultural = st_data['Cultural Accuracy & Authenticity'].mean()
        
        results.append({
            'story_type': story_type,
            'n_stories': len(st_data),
            'mean_genericity': mean_genericity,
            'std_genericity': std_genericity,
            'pct_generic': pct_generic,
            'mean_cultural_accuracy': mean_cultural
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_genericity', ascending=False)
    
    return results_df


def compute_genericity_deciles(df):
    """Compute quality statistics by genericity decile."""
    df = df.copy()
    
    df['genericity_decile'] = pd.qcut(df['genericity_score'], q=10, labels=False, duplicates='drop')
    
    results = []
    for decile in sorted(df['genericity_decile'].unique()):
        decile_data = df[df['genericity_decile'] == decile]
        
        row = {
            'decile': decile + 1,
            'n_stories': len(decile_data),
            'mean_genericity': decile_data['genericity_score'].mean(),
            'min_genericity': decile_data['genericity_score'].min(),
            'max_genericity': decile_data['genericity_score'].max()
        }
        
        for metric in TARGET_METRICS + ['average_score']:
            short_name = METRIC_SHORT_NAMES.get(metric, 'average_score')
            row[f'mean_{short_name}'] = decile_data[metric].mean()
        
        results.append(row)
    
    return pd.DataFrame(results)


def create_visualizations(df, comparison, correlations, model_analysis, 
                         culture_analysis, decile_stats, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    cultural_col = 'Cultural Accuracy & Authenticity'
    
    # 1. Scatter plot: Genericity vs Cultural Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if g else 'blue' for g in df['is_generic']]
    scatter = ax.scatter(df['genericity_score'], df[cultural_col], 
                         alpha=0.5, c=colors, edgecolors='black', linewidth=0.5, s=50)
    
    # Add trend line
    z = np.polyfit(df['genericity_score'], df[cultural_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['genericity_score'].min(), df['genericity_score'].max(), 100)
    corr_val = correlations[correlations['metric_short'] == 'cultural_accuracy']['pearson_r'].values[0]
    ax.plot(x_line, p(x_line), 'g--', linewidth=2, label=f'Trend (r={corr_val:.3f})')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Non-generic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Generic (top 10%)'),
        Line2D([0], [0], color='g', linestyle='--', linewidth=2, label=f'Trend (r={corr_val:.3f})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('Genericity Score (similarity to centroid)', fontsize=11)
    ax.set_ylabel('Cultural Accuracy Score', fontsize=11)
    ax.set_title('RQ14: Genericity vs. Cultural Accuracy', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_genericity_vs_cultural.png', bbox_inches='tight')
    plt.close()
    
    # 2. Comparison bar chart: Generic vs Non-generic
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['generic_mean'], width, label='Generic', color='salmon', edgecolor='black')
    bars2 = ax.bar(x + width/2, comparison['nongeneric_mean'], width, label='Non-generic', color='lightgreen', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['metric_short'], rotation=45, ha='right')
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ14: Quality Scores - Generic vs. Non-generic Stories', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars2, comparison['significant'])):
        if sig:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, '*', 
                   ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_generic_vs_nongeneric.png', bbox_inches='tight')
    plt.close()
    
    # 3. Genericity by model (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_sorted = model_analysis.sort_values('mean_genericity', ascending=False)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_sorted)))
    
    bars = ax.bar(range(len(model_sorted)), model_sorted['mean_genericity'], 
                  yerr=model_sorted['std_genericity'], capsize=5, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(model_sorted)))
    ax.set_xticklabels(model_sorted['model'], rotation=45, ha='right')
    ax.set_ylabel('Mean Genericity Score', fontsize=11)
    ax.set_title('RQ14: Genericity by Generation Model\n(Higher = More Generic/Safe)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, model_sorted['mean_genericity']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_genericity_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 4. Box plot: Quality by generic status
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(TARGET_METRICS):
        ax = axes[idx]
        short_name = METRIC_SHORT_NAMES[metric]
        
        plot_data = df[['is_generic', metric]].copy()
        plot_data['Status'] = plot_data['is_generic'].map({True: 'Generic', False: 'Non-generic'})
        
        sns.boxplot(data=plot_data, x='Status', y=metric, 
                   palette={'Generic': 'salmon', 'Non-generic': 'lightgreen'}, ax=ax)
        ax.set_title(f'{short_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
    
    plt.suptitle('RQ14: Score Distributions by Generic Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_boxplots.png', bbox_inches='tight')
    plt.close()
    
    # 5. Decile analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(decile_stats['decile'], decile_stats['mean_cultural_accuracy'], 'o-', 
            color='steelblue', markersize=10, linewidth=2, label='Cultural Accuracy')
    ax.plot(decile_stats['decile'], decile_stats['mean_average_score'], 's--', 
            color='coral', markersize=10, linewidth=2, label='Average Score')
    
    ax.set_xlabel('Genericity Decile (1=Least Generic, 10=Most Generic)', fontsize=11)
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ14: Quality Scores by Genericity Decile', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xticks(decile_stats['decile'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_decile_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap: Model × Culture genericity
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot = df.pivot_table(
        values='genericity_score',
        index='model',
        columns='culture',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Mean Genericity Score'})
    ax.set_title('RQ14: Mean Genericity by Model × Culture', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_heatmap_model_culture.png', bbox_inches='tight')
    plt.close()
    
    # 7. Genericity penalty visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    penalties = comparison[comparison['metric_short'] != 'average_score'][['metric_short', 'genericity_penalty', 'significant']]
    colors = ['green' if p > 0 else 'red' for p in penalties['genericity_penalty']]
    
    bars = ax.bar(penalties['metric_short'], penalties['genericity_penalty'], color=colors, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Genericity Penalty\n(Non-generic - Generic Score)', fontsize=11)
    ax.set_title('RQ14: Genericity Penalty by Metric\n(Positive = Generic stories score LOWER)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add significance markers
    for bar, sig in zip(bars, penalties['significant']):
        if sig:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height >= 0 else height - 0.05,
                   '*', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq14_genericity_penalty.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, comparison, correlations, model_analysis, generic_threshold):
    """Generate summary of key findings."""
    # Cultural accuracy findings
    cultural_corr = correlations[correlations['metric_short'] == 'cultural_accuracy'].iloc[0]
    cultural_comparison = comparison[comparison['metric_short'] == 'cultural_accuracy'].iloc[0]
    avg_comparison = comparison[comparison['metric_short'] == 'average_score'].iloc[0]
    
    # Model findings
    most_generic_model = model_analysis.iloc[0]
    least_generic_model = model_analysis.iloc[-1]
    
    summary = {
        'research_question': 'RQ14: The Safety Filter Effect',
        'main_question': 'Do generic/safe outputs receive lower cultural scores?',
        'n_stories': len(df),
        'n_generic': df['is_generic'].sum(),
        'pct_generic': df['is_generic'].mean() * 100,
        'generic_threshold': generic_threshold,
        
        # Correlation findings
        'genericity_cultural_correlation': cultural_corr['pearson_r'],
        'genericity_cultural_p_value': cultural_corr['pearson_p_value'],
        'genericity_cultural_significant': cultural_corr['pearson_significant'],
        
        # Comparison findings
        'generic_cultural_mean': cultural_comparison['generic_mean'],
        'nongeneric_cultural_mean': cultural_comparison['nongeneric_mean'],
        'cultural_penalty': cultural_comparison['genericity_penalty'],
        'cultural_penalty_significant': cultural_comparison['significant'],
        
        'generic_avg_score': avg_comparison['generic_mean'],
        'nongeneric_avg_score': avg_comparison['nongeneric_mean'],
        'avg_score_penalty': avg_comparison['genericity_penalty'],
        'avg_score_penalty_significant': avg_comparison['significant'],
        
        # Model findings
        'most_generic_model': most_generic_model['model'],
        'most_generic_score': most_generic_model['mean_genericity'],
        'most_generic_pct': most_generic_model['pct_generic'],
        'least_generic_model': least_generic_model['model'],
        'least_generic_score': least_generic_model['mean_genericity'],
        
        # Interpretation
        'safety_filter_effect_exists': cultural_comparison['significant'] and cultural_comparison['genericity_penalty'] > 0
    }
    
    return summary


def main():
    """Main execution function for RQ14 analysis."""
    print("=" * 70)
    print("RQ14: The 'Safety Filter' Effect")
    print("=" * 70)
    print("\nQuestion: Do generic/safe outputs receive lower cultural scores?")
    print("Method: Analyze relationship between genericity and quality scores")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading and merging data...")
    df = load_data()
    print(f"    Loaded {len(df)} stories")
    
    # Compute overall centroid
    print("\n[2] Computing overall embedding centroid...")
    centroid = compute_overall_centroid(df)
    print(f"    Centroid dimension: {len(centroid)}")
    
    # Compute genericity
    print("\n[3] Computing genericity scores...")
    df = compute_genericity(df, centroid)
    print(f"    Genericity range: {df['genericity_score'].min():.4f} - {df['genericity_score'].max():.4f}")
    print(f"    Mean genericity: {df['genericity_score'].mean():.4f} (±{df['genericity_score'].std():.4f})")
    
    # Identify generic stories
    print("\n[4] Identifying generic stories (top 10%)...")
    df, threshold = identify_generic_stories(df, percentile=10)
    n_generic = df['is_generic'].sum()
    print(f"    Generic threshold: {threshold:.4f}")
    print(f"    Generic stories: {n_generic} ({n_generic/len(df)*100:.1f}%)")
    
    # Compare generic vs non-generic
    print("\n[5] Comparing generic vs. non-generic scores...")
    comparison = compare_generic_vs_nongeneric(df)
    comparison.to_csv(output_dir / 'rq14_generic_comparison.csv', index=False)
    print(f"    Exported: rq14_generic_comparison.csv")
    
    print("\n    Quality Comparison:")
    for _, row in comparison.iterrows():
        sig = "*" if row['significant'] else ""
        penalty_dir = "⬇" if row['genericity_penalty'] > 0 else "⬆"
        print(f"      {row['metric_short']}: Generic={row['generic_mean']:.3f}, "
              f"Non-generic={row['nongeneric_mean']:.3f}, Penalty={row['genericity_penalty']:.3f} {penalty_dir} {sig}")
    
    # Correlate with scores
    print("\n[6] Correlating genericity with scores...")
    correlations = correlate_genericity_with_scores(df)
    correlations.to_csv(output_dir / 'rq14_genericity_correlations.csv', index=False)
    print(f"    Exported: rq14_genericity_correlations.csv")
    
    print("\n    Genericity-Score Correlations:")
    for _, row in correlations.iterrows():
        sig = "*" if row['pearson_significant'] else ""
        print(f"      {row['metric_short']}: r = {row['pearson_r']:.4f} {sig}")
    
    # Model analysis
    print("\n[7] Analyzing genericity by model...")
    model_analysis = analyze_genericity_by_model(df)
    model_analysis.to_csv(output_dir / 'rq14_genericity_by_model.csv', index=False)
    print(f"    Exported: rq14_genericity_by_model.csv")
    
    print("\n    Model Genericity Ranking (Most → Least Generic):")
    for _, row in model_analysis.iterrows():
        print(f"      {row['model']}: {row['mean_genericity']:.4f} ({row['pct_generic']:.1f}% generic)")
    
    # Culture analysis
    print("\n[8] Analyzing genericity by culture...")
    culture_analysis = analyze_genericity_by_culture(df)
    culture_analysis.to_csv(output_dir / 'rq14_genericity_by_culture.csv', index=False)
    print(f"    Exported: rq14_genericity_by_culture.csv")
    
    # Story type analysis
    print("\n[9] Analyzing genericity by story type...")
    story_type_analysis = analyze_genericity_by_story_type(df)
    story_type_analysis.to_csv(output_dir / 'rq14_genericity_by_story_type.csv', index=False)
    print(f"    Exported: rq14_genericity_by_story_type.csv")
    
    # Decile analysis
    print("\n[10] Computing decile statistics...")
    decile_stats = compute_genericity_deciles(df)
    decile_stats.to_csv(output_dir / 'rq14_genericity_deciles.csv', index=False)
    print(f"    Exported: rq14_genericity_deciles.csv")
    
    # Export detailed data
    print("\n[11] Exporting detailed story data...")
    export_cols = ['model', 'culture', 'story_type', 'region', 
                   'genericity_score', 'centroid_similarity', 'is_generic', 'average_score'] + TARGET_METRICS
    df[export_cols].to_csv(output_dir / 'rq14_story_genericity_data.csv', index=False)
    print(f"    Exported: rq14_story_genericity_data.csv")
    
    # Create visualizations
    print("\n[12] Creating visualizations...")
    create_visualizations(df, comparison, correlations, model_analysis,
                         culture_analysis, decile_stats, output_dir)
    
    # Generate summary
    print("\n[13] Generating summary...")
    summary = generate_summary(df, comparison, correlations, model_analysis, threshold)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq14_summary.csv', index=False)
    print(f"    Exported: rq14_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ14 ANALYSIS COMPLETE")
    print("=" * 70)
    
    cultural_result = comparison[comparison['metric_short'] == 'cultural_accuracy'].iloc[0]
    avg_result = comparison[comparison['metric_short'] == 'average_score'].iloc[0]
    
    print(f"\nKey Findings:")
    
    print(f"\n  Genericity Penalty (Cultural Accuracy):")
    if cultural_result['significant'] and cultural_result['genericity_penalty'] > 0:
        print(f"    ✓ SIGNIFICANT penalty exists!")
        print(f"    Generic stories score {cultural_result['genericity_penalty']:.3f} LOWER than non-generic")
        print(f"    → Playing it safe semantically hurts cultural authenticity")
    elif cultural_result['genericity_penalty'] > 0:
        print(f"    ⚠️  Penalty exists but not statistically significant")
        print(f"    Generic: {cultural_result['generic_mean']:.3f}, Non-generic: {cultural_result['nongeneric_mean']:.3f}")
    else:
        print(f"    ✗ No penalty detected")
        print(f"    Generic stories score similar or higher than non-generic")
    
    print(f"\n  Model Genericity Ranking:")
    print(f"    Most Generic:  {model_analysis.iloc[0]['model']} ({model_analysis.iloc[0]['mean_genericity']:.4f})")
    print(f"    Least Generic: {model_analysis.iloc[-1]['model']} ({model_analysis.iloc[-1]['mean_genericity']:.4f})")
    
    # Overall conclusion
    if summary['safety_filter_effect_exists']:
        print(f"\n  ✓ CONCLUSION: Safety filter effect CONFIRMED")
        print(f"    Generic/safe outputs receive lower cultural scores.")
        print(f"    LLMs that produce more unique content score higher on cultural authenticity.")
    else:
        print(f"\n  ✗ CONCLUSION: No clear safety filter effect")
        print(f"    Genericity does not significantly predict lower cultural scores.")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq14_generic_comparison.csv - Generic vs non-generic comparison")
    print("  2. rq14_genericity_correlations.csv - Genericity-score correlations")
    print("  3. rq14_genericity_by_model.csv - Model genericity analysis")
    print("  4. rq14_genericity_by_culture.csv - Culture genericity analysis")
    print("  5. rq14_genericity_by_story_type.csv - Story type analysis")
    print("  6. rq14_genericity_deciles.csv - Decile statistics")
    print("  7. rq14_story_genericity_data.csv - Per-story data")
    print("  8. rq14_summary.csv - Key findings")
    print("-" * 70)
    
    return df, comparison, model_analysis


if __name__ == "__main__":
    df, comparison, model_analysis = main()
