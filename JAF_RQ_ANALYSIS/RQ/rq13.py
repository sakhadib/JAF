"""
RQ13: Semantic Drift vs. Human Perception
=========================================
Question: Does prompt-story divergence predict coherence scores?
          Is there a threshold where semantic drift leads to quality drop?

Method:
- Compute semantic drift: 1 - cosine_similarity(scenario_emb, story_emb)
- Correlate drift with Narrative & Symbolic Coherence score
- Visualize: scatter plot of drift vs. coherence
- Threshold analysis: identify drift threshold indicating quality drop

Data Source: Parquet (scenario_embedding, story_embedding) + CSV (scores)

Interpretation:
- High drift = story diverged significantly from prompt
- If drift correlates negatively with coherence → staying on-topic matters
- Threshold analysis helps identify "acceptable" drift levels
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
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
        parquet_df[['model', 'culture', 'story', 'story_type', 'region', 
                    'story_embedding', 'scenario_embedding']],
        on=['model', 'culture', 'story'],
        how='inner'
    )
    
    return merged


def compute_semantic_drift(df):
    """
    Compute semantic drift: 1 - cosine_similarity(scenario_emb, story_emb)
    Higher drift = story diverged more from the prompt/scenario
    """
    df = df.copy()
    
    drifts = []
    similarities = []
    
    for idx, row in df.iterrows():
        scenario_emb = np.array(row['scenario_embedding'])
        story_emb = np.array(row['story_embedding'])
        
        # Cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(scenario_emb, story_emb)
        drift = 1 - similarity  # Same as cosine distance
        
        similarities.append(similarity)
        drifts.append(drift)
    
    df['prompt_story_similarity'] = similarities
    df['semantic_drift'] = drifts
    
    # Compute average score
    df['average_score'] = df[TARGET_METRICS].mean(axis=1)
    
    return df


def correlate_drift_with_scores(df):
    """Compute correlation between semantic drift and all quality scores."""
    results = []
    
    for metric in TARGET_METRICS:
        short_name = METRIC_SHORT_NAMES[metric]
        
        # Pearson correlation
        r, p = stats.pearsonr(df['semantic_drift'], df[metric])
        
        # Spearman correlation (rank-based, more robust)
        rho, p_spearman = stats.spearmanr(df['semantic_drift'], df[metric])
        
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
    
    # Also with average score
    r, p = stats.pearsonr(df['semantic_drift'], df['average_score'])
    rho, p_spearman = stats.spearmanr(df['semantic_drift'], df['average_score'])
    
    results.append({
        'metric': 'Average Score',
        'metric_short': 'average_score',
        'pearson_r': r,
        'pearson_p_value': p,
        'pearson_significant': p < 0.05,
        'spearman_rho': rho,
        'spearman_p_value': p_spearman,
        'spearman_significant': p_spearman < 0.05
    })
    
    return pd.DataFrame(results)


def analyze_drift_by_model(df):
    """Analyze semantic drift patterns by generation model."""
    results = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Drift statistics
        mean_drift = model_data['semantic_drift'].mean()
        std_drift = model_data['semantic_drift'].std()
        median_drift = model_data['semantic_drift'].median()
        
        # Correlation with coherence for this model
        coherence_col = 'Narrative & Symbolic Coherence'
        r, p = stats.pearsonr(model_data['semantic_drift'], model_data[coherence_col])
        
        results.append({
            'model': model,
            'n_stories': len(model_data),
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'median_drift': median_drift,
            'min_drift': model_data['semantic_drift'].min(),
            'max_drift': model_data['semantic_drift'].max(),
            'drift_coherence_correlation': r,
            'correlation_p_value': p,
            'correlation_significant': p < 0.05,
            'mean_coherence_score': model_data[coherence_col].mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_drift')
    
    return results_df


def analyze_drift_by_culture(df):
    """Analyze semantic drift patterns by culture."""
    results = []
    
    coherence_col = 'Narrative & Symbolic Coherence'
    
    for culture in df['culture'].unique():
        culture_data = df[df['culture'] == culture]
        
        mean_drift = culture_data['semantic_drift'].mean()
        std_drift = culture_data['semantic_drift'].std()
        
        r, p = stats.pearsonr(culture_data['semantic_drift'], culture_data[coherence_col])
        
        results.append({
            'culture': culture,
            'n_stories': len(culture_data),
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'drift_coherence_correlation': r,
            'correlation_p_value': p,
            'correlation_significant': p < 0.05,
            'mean_coherence_score': culture_data[coherence_col].mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_drift')
    
    return results_df


def analyze_drift_by_story_type(df):
    """Analyze semantic drift patterns by story type."""
    results = []
    
    coherence_col = 'Narrative & Symbolic Coherence'
    
    for story_type in df['story_type'].unique():
        st_data = df[df['story_type'] == story_type]
        
        mean_drift = st_data['semantic_drift'].mean()
        std_drift = st_data['semantic_drift'].std()
        
        r, p = stats.pearsonr(st_data['semantic_drift'], st_data[coherence_col])
        
        results.append({
            'story_type': story_type,
            'n_stories': len(st_data),
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'drift_coherence_correlation': r,
            'correlation_p_value': p,
            'correlation_significant': p < 0.05,
            'mean_coherence_score': st_data[coherence_col].mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_drift')
    
    return results_df


def perform_threshold_analysis(df, coherence_col='Narrative & Symbolic Coherence'):
    """
    Identify drift thresholds that indicate quality drop.
    Uses decile analysis and change point detection.
    """
    df = df.copy()
    
    # Create drift deciles
    df['drift_decile'] = pd.qcut(df['semantic_drift'], q=10, labels=False, duplicates='drop')
    
    decile_stats = []
    for decile in sorted(df['drift_decile'].unique()):
        decile_data = df[df['drift_decile'] == decile]
        
        decile_stats.append({
            'decile': decile + 1,
            'drift_min': decile_data['semantic_drift'].min(),
            'drift_max': decile_data['semantic_drift'].max(),
            'drift_mean': decile_data['semantic_drift'].mean(),
            'n_stories': len(decile_data),
            'mean_coherence': decile_data[coherence_col].mean(),
            'std_coherence': decile_data[coherence_col].std(),
            'mean_avg_score': decile_data['average_score'].mean()
        })
    
    decile_df = pd.DataFrame(decile_stats)
    
    # Find threshold where quality drops significantly
    # Compare each decile to the first (lowest drift)
    baseline_coherence = decile_df.iloc[0]['mean_coherence']
    
    threshold_results = []
    for idx, row in decile_df.iterrows():
        drop_from_baseline = baseline_coherence - row['mean_coherence']
        pct_drop = (drop_from_baseline / baseline_coherence) * 100 if baseline_coherence > 0 else 0
        
        threshold_results.append({
            'decile': row['decile'],
            'drift_threshold': row['drift_max'],
            'mean_coherence': row['mean_coherence'],
            'drop_from_baseline': drop_from_baseline,
            'pct_drop': pct_drop,
            'significant_drop': pct_drop > 5  # 5% drop threshold
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    # Find first significant drop
    significant_drops = threshold_df[threshold_df['significant_drop']]
    if len(significant_drops) > 0:
        critical_threshold = significant_drops.iloc[0]['drift_threshold']
    else:
        critical_threshold = None
    
    return decile_df, threshold_df, critical_threshold


def compute_drift_quality_bins(df):
    """Bin stories by drift level and compute quality statistics."""
    df = df.copy()
    
    # Create drift bins
    bins = [0, 0.2, 0.3, 0.4, 0.5, 1.0]
    labels = ['Very Low (0-0.2)', 'Low (0.2-0.3)', 'Medium (0.3-0.4)', 'High (0.4-0.5)', 'Very High (0.5+)']
    
    df['drift_bin'] = pd.cut(df['semantic_drift'], bins=bins, labels=labels, include_lowest=True)
    
    results = []
    for bin_label in labels:
        bin_data = df[df['drift_bin'] == bin_label]
        
        if len(bin_data) == 0:
            continue
        
        row = {
            'drift_bin': bin_label,
            'n_stories': len(bin_data),
            'mean_drift': bin_data['semantic_drift'].mean(),
            'mean_average_score': bin_data['average_score'].mean()
        }
        
        for metric in TARGET_METRICS:
            short_name = METRIC_SHORT_NAMES[metric]
            row[f'mean_{short_name}'] = bin_data[metric].mean()
        
        results.append(row)
    
    return pd.DataFrame(results)


def create_visualizations(df, correlations, model_analysis, culture_analysis, 
                         story_type_analysis, decile_df, drift_bins, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    coherence_col = 'Narrative & Symbolic Coherence'
    
    # 1. Main scatter plot: Drift vs Coherence
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df['semantic_drift'], df[coherence_col], 
                         alpha=0.5, c=df['average_score'], cmap='RdYlGn',
                         edgecolors='black', linewidth=0.5, s=50)
    
    # Add trend line
    z = np.polyfit(df['semantic_drift'], df[coherence_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['semantic_drift'].min(), df['semantic_drift'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
            label=f'Trend (r={correlations[correlations["metric_short"]=="narrative_coherence"]["pearson_r"].values[0]:.3f})')
    
    ax.set_xlabel('Semantic Drift (1 - cosine similarity)', fontsize=11)
    ax.set_ylabel('Narrative Coherence Score', fontsize=11)
    ax.set_title('RQ13: Semantic Drift vs. Narrative Coherence', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Average Score')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_vs_coherence.png', bbox_inches='tight')
    plt.close()
    
    # 2. Correlation bar chart for all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(correlations))
    colors = ['green' if r < 0 else 'red' for r in correlations['pearson_r']]
    
    bars = ax.bar(x, correlations['pearson_r'], color=colors, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(correlations['metric_short'], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation (r)', fontsize=11)
    ax.set_title('RQ13: Drift-Score Correlation by Metric', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, correlations['pearson_significant'])):
        if sig:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), '*', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_correlations.png', bbox_inches='tight')
    plt.close()
    
    # 3. Drift by model (box plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_order = model_analysis.sort_values('mean_drift')['model'].tolist()
    sns.boxplot(data=df, x='model', y='semantic_drift', order=model_order, palette='viridis', ax=ax)
    
    ax.set_xlabel('Generation Model', fontsize=11)
    ax.set_ylabel('Semantic Drift', fontsize=11)
    ax.set_title('RQ13: Semantic Drift Distribution by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 4. Decile analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.bar(decile_df['decile'], decile_df['mean_coherence'], 
           color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(decile_df))), edgecolor='black')
    ax.set_xlabel('Drift Decile (1=Lowest, 10=Highest)', fontsize=11)
    ax.set_ylabel('Mean Coherence Score', fontsize=11)
    ax.set_title('Coherence by Drift Decile', fontsize=12, fontweight='bold')
    
    ax = axes[1]
    ax.plot(decile_df['drift_mean'], decile_df['mean_coherence'], 'o-', 
            color='steelblue', markersize=10, linewidth=2)
    ax.set_xlabel('Mean Drift', fontsize=11)
    ax.set_ylabel('Mean Coherence Score', fontsize=11)
    ax.set_title('Coherence vs Drift (Decile Means)', fontsize=12, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(decile_df['drift_mean'], decile_df['mean_coherence'], 1)
    p = np.poly1d(z)
    ax.plot(decile_df['drift_mean'], p(decile_df['drift_mean']), 'r--', linewidth=2, label='Linear Trend')
    ax.legend()
    
    plt.suptitle('RQ13: Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_threshold_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 5. Drift bins quality comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(drift_bins))
    width = 0.15
    
    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
    
    for i, metric in enumerate(TARGET_METRICS):
        short_name = METRIC_SHORT_NAMES[metric]
        col_name = f'mean_{short_name}'
        if col_name in drift_bins.columns:
            bars = ax.bar(x + i*width, drift_bins[col_name], width, label=short_name, color=colors[i])
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(drift_bins['drift_bin'], rotation=45, ha='right')
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ13: Quality Scores by Drift Level', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_bins_quality.png', bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap: Model × Culture drift
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot = df.pivot_table(
        values='semantic_drift',
        index='model',
        columns='culture',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Mean Semantic Drift'})
    ax.set_title('RQ13: Mean Semantic Drift by Model × Culture', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 7. Scatter plots by model (colored)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax.scatter(model_data['semantic_drift'], model_data[coherence_col], 
                   alpha=0.5, label=model, s=50)
    
    ax.set_xlabel('Semantic Drift', fontsize=11)
    ax.set_ylabel('Narrative Coherence', fontsize=11)
    ax.set_title('RQ13: Drift vs Coherence by Model', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq13_drift_vs_coherence_by_model.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, correlations, model_analysis, critical_threshold):
    """Generate summary of key findings."""
    coherence_corr = correlations[correlations['metric_short'] == 'narrative_coherence'].iloc[0]
    avg_corr = correlations[correlations['metric_short'] == 'average_score'].iloc[0]
    
    # Model with lowest/highest drift
    lowest_drift_model = model_analysis.iloc[0]
    highest_drift_model = model_analysis.iloc[-1]
    
    summary = {
        'research_question': 'RQ13: Semantic Drift vs. Human Perception',
        'main_question': 'Does prompt-story divergence predict coherence scores?',
        'n_stories': len(df),
        
        # Drift statistics
        'mean_drift': df['semantic_drift'].mean(),
        'std_drift': df['semantic_drift'].std(),
        'min_drift': df['semantic_drift'].min(),
        'max_drift': df['semantic_drift'].max(),
        
        # Coherence correlation
        'drift_coherence_pearson_r': coherence_corr['pearson_r'],
        'drift_coherence_p_value': coherence_corr['pearson_p_value'],
        'drift_coherence_significant': coherence_corr['pearson_significant'],
        
        # Average score correlation
        'drift_avg_score_pearson_r': avg_corr['pearson_r'],
        'drift_avg_score_p_value': avg_corr['pearson_p_value'],
        'drift_avg_score_significant': avg_corr['pearson_significant'],
        
        # Model analysis
        'lowest_drift_model': lowest_drift_model['model'],
        'lowest_drift_value': lowest_drift_model['mean_drift'],
        'highest_drift_model': highest_drift_model['model'],
        'highest_drift_value': highest_drift_model['mean_drift'],
        
        # Threshold
        'critical_drift_threshold': critical_threshold,
        
        # Interpretation
        'drift_predicts_coherence': coherence_corr['pearson_r'] < -0.1 and coherence_corr['pearson_significant']
    }
    
    return summary


def main():
    """Main execution function for RQ13 analysis."""
    print("=" * 70)
    print("RQ13: Semantic Drift vs. Human Perception")
    print("=" * 70)
    print("\nQuestion: Does prompt-story divergence predict coherence scores?")
    print("Method: Correlate semantic drift with narrative coherence")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading and merging data...")
    df = load_data()
    print(f"    Loaded {len(df)} stories")
    
    # Compute semantic drift
    print("\n[2] Computing semantic drift...")
    df = compute_semantic_drift(df)
    print(f"    Drift range: {df['semantic_drift'].min():.4f} - {df['semantic_drift'].max():.4f}")
    print(f"    Mean drift: {df['semantic_drift'].mean():.4f} (±{df['semantic_drift'].std():.4f})")
    
    # Correlate with scores
    print("\n[3] Correlating drift with quality scores...")
    correlations = correlate_drift_with_scores(df)
    correlations.to_csv(output_dir / 'rq13_drift_correlations.csv', index=False)
    print(f"    Exported: rq13_drift_correlations.csv")
    
    print("\n    Drift-Score Correlations:")
    for _, row in correlations.iterrows():
        sig = "*" if row['pearson_significant'] else ""
        print(f"      {row['metric_short']}: r = {row['pearson_r']:.4f} {sig}")
    
    # Model analysis
    print("\n[4] Analyzing drift by model...")
    model_analysis = analyze_drift_by_model(df)
    model_analysis.to_csv(output_dir / 'rq13_drift_by_model.csv', index=False)
    print(f"    Exported: rq13_drift_by_model.csv")
    
    print("\n    Mean Drift by Model:")
    for _, row in model_analysis.iterrows():
        print(f"      {row['model']}: {row['mean_drift']:.4f}")
    
    # Culture analysis
    print("\n[5] Analyzing drift by culture...")
    culture_analysis = analyze_drift_by_culture(df)
    culture_analysis.to_csv(output_dir / 'rq13_drift_by_culture.csv', index=False)
    print(f"    Exported: rq13_drift_by_culture.csv")
    
    # Story type analysis
    print("\n[6] Analyzing drift by story type...")
    story_type_analysis = analyze_drift_by_story_type(df)
    story_type_analysis.to_csv(output_dir / 'rq13_drift_by_story_type.csv', index=False)
    print(f"    Exported: rq13_drift_by_story_type.csv")
    
    # Threshold analysis
    print("\n[7] Performing threshold analysis...")
    decile_df, threshold_df, critical_threshold = perform_threshold_analysis(df)
    decile_df.to_csv(output_dir / 'rq13_drift_deciles.csv', index=False)
    threshold_df.to_csv(output_dir / 'rq13_threshold_analysis.csv', index=False)
    print(f"    Exported: rq13_drift_deciles.csv, rq13_threshold_analysis.csv")
    
    if critical_threshold:
        print(f"    Critical drift threshold: {critical_threshold:.4f}")
    else:
        print(f"    No critical threshold detected (quality stable across drift levels)")
    
    # Drift bins
    print("\n[8] Computing drift bin statistics...")
    drift_bins = compute_drift_quality_bins(df)
    drift_bins.to_csv(output_dir / 'rq13_drift_bins.csv', index=False)
    print(f"    Exported: rq13_drift_bins.csv")
    
    # Export detailed story data
    print("\n[9] Exporting detailed story data...")
    export_cols = ['model', 'culture', 'story_type', 'region', 
                   'semantic_drift', 'prompt_story_similarity', 'average_score'] + TARGET_METRICS
    df[export_cols].to_csv(output_dir / 'rq13_story_drift_data.csv', index=False)
    print(f"    Exported: rq13_story_drift_data.csv")
    
    # Create visualizations
    print("\n[10] Creating visualizations...")
    create_visualizations(df, correlations, model_analysis, culture_analysis,
                         story_type_analysis, decile_df, drift_bins, output_dir)
    
    # Generate summary
    print("\n[11] Generating summary...")
    summary = generate_summary(df, correlations, model_analysis, critical_threshold)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq13_summary.csv', index=False)
    print(f"    Exported: rq13_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ13 ANALYSIS COMPLETE")
    print("=" * 70)
    
    coherence_corr = correlations[correlations['metric_short'] == 'narrative_coherence'].iloc[0]
    
    print(f"\nKey Findings:")
    
    print(f"\n  Drift-Coherence Correlation:")
    if coherence_corr['pearson_r'] < -0.1 and coherence_corr['pearson_significant']:
        print(f"    ✓ NEGATIVE correlation (r = {coherence_corr['pearson_r']:.4f}, p < 0.05)")
        print(f"    → Stories that drift from prompt have LOWER coherence scores")
    elif coherence_corr['pearson_r'] < 0:
        print(f"    ⚠️  Weak negative correlation (r = {coherence_corr['pearson_r']:.4f})")
        print(f"    → Slight tendency for drifting stories to score lower")
    else:
        print(f"    ✗ No meaningful correlation (r = {coherence_corr['pearson_r']:.4f})")
        print(f"    → Semantic drift does not predict coherence")
    
    print(f"\n  Model Drift Patterns:")
    print(f"    Lowest drift:  {model_analysis.iloc[0]['model']} ({model_analysis.iloc[0]['mean_drift']:.4f})")
    print(f"    Highest drift: {model_analysis.iloc[-1]['model']} ({model_analysis.iloc[-1]['mean_drift']:.4f})")
    
    if critical_threshold:
        print(f"\n  Threshold Analysis:")
        print(f"    Critical threshold: {critical_threshold:.4f}")
        print(f"    → Quality drops significantly when drift exceeds this value")
    
    # Check all metric correlations
    all_negative = all(correlations['pearson_r'] < 0)
    all_significant = all(correlations['pearson_significant'])
    
    if all_negative and all_significant:
        print(f"\n  ✓ CONCLUSION: Semantic drift negatively impacts ALL quality metrics")
        print(f"    Stories that stay closer to the prompt score higher across the board.")
    elif all_negative:
        print(f"\n  ⚠️  CONCLUSION: Drift tends to reduce quality, but effect varies by metric")
    else:
        print(f"\n  ✗ CONCLUSION: Mixed relationship between drift and quality")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq13_drift_correlations.csv - Drift-score correlations")
    print("  2. rq13_drift_by_model.csv - Drift statistics by model")
    print("  3. rq13_drift_by_culture.csv - Drift statistics by culture")
    print("  4. rq13_drift_by_story_type.csv - Drift statistics by story type")
    print("  5. rq13_drift_deciles.csv - Decile analysis")
    print("  6. rq13_threshold_analysis.csv - Threshold detection")
    print("  7. rq13_drift_bins.csv - Quality by drift bin")
    print("  8. rq13_story_drift_data.csv - Per-story drift data")
    print("  9. rq13_summary.csv - Key findings")
    print("-" * 70)
    
    return df, correlations, model_analysis


if __name__ == "__main__":
    df, correlations, model_analysis = main()
