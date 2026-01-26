"""
RQ12: Identifying Hallucination in Vector Space
================================================
Question: Are low-quality stories outliers in embedding space?
          Can we detect poor quality/hallucinated content via embedding analysis?

Method:
- Define "ideal" centroid: Mean embedding of high-quality stories (score >= 4)
- Compute distance of each story from ideal centroid
- Correlate distances with human evaluation scores
- Apply outlier detection (Isolation Forest, LOF) on embeddings
- Analyze: Do outliers correlate with low quality scores?

Data Source: Parquet (story_embedding) + CSV (evaluation scores)

Interpretation:
- High distance from ideal = potential hallucination/low quality
- Outlier status should correlate with low scores if embeddings capture quality
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")

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


def compute_average_score(df):
    """Compute average score across all metrics for each story."""
    df = df.copy()
    df['average_score'] = df[TARGET_METRICS].mean(axis=1)
    return df


def define_ideal_centroid(df, threshold=4.0):
    """
    Define ideal centroid as mean embedding of high-quality stories.
    High quality defined as average score >= threshold.
    """
    high_quality_mask = df['average_score'] >= threshold
    high_quality_stories = df[high_quality_mask]
    
    if len(high_quality_stories) == 0:
        # Fallback: use top 20% by score
        score_threshold = df['average_score'].quantile(0.8)
        high_quality_mask = df['average_score'] >= score_threshold
        high_quality_stories = df[high_quality_mask]
        print(f"    No stories with score >= {threshold}. Using top 20% (threshold: {score_threshold:.2f})")
    
    embeddings = np.array(high_quality_stories['story_embedding'].tolist())
    ideal_centroid = embeddings.mean(axis=0)
    
    return ideal_centroid, len(high_quality_stories)


def compute_distances_from_ideal(df, ideal_centroid):
    """Compute distance of each story from the ideal centroid."""
    df = df.copy()
    
    embeddings = np.array(df['story_embedding'].tolist())
    
    # Cosine distance
    cosine_distances = []
    for emb in embeddings:
        cosine_distances.append(cosine(emb, ideal_centroid))
    
    # Euclidean distance
    euclidean_distances = []
    for emb in embeddings:
        euclidean_distances.append(euclidean(emb, ideal_centroid))
    
    df['cosine_distance_from_ideal'] = cosine_distances
    df['euclidean_distance_from_ideal'] = euclidean_distances
    
    return df


def correlate_distance_with_scores(df):
    """Compute correlation between distance from ideal and quality scores."""
    results = []
    
    for metric in TARGET_METRICS:
        short_name = METRIC_SHORT_NAMES[metric]
        
        # Cosine distance correlation
        r_cos, p_cos = stats.pearsonr(df['cosine_distance_from_ideal'], df[metric])
        
        # Euclidean distance correlation
        r_euc, p_euc = stats.pearsonr(df['euclidean_distance_from_ideal'], df[metric])
        
        results.append({
            'metric': metric,
            'metric_short': short_name,
            'cosine_correlation': r_cos,
            'cosine_p_value': p_cos,
            'cosine_significant': p_cos < 0.05,
            'euclidean_correlation': r_euc,
            'euclidean_p_value': p_euc,
            'euclidean_significant': p_euc < 0.05
        })
    
    # Also correlate with average score
    r_cos_avg, p_cos_avg = stats.pearsonr(df['cosine_distance_from_ideal'], df['average_score'])
    r_euc_avg, p_euc_avg = stats.pearsonr(df['euclidean_distance_from_ideal'], df['average_score'])
    
    results.append({
        'metric': 'Average Score',
        'metric_short': 'average_score',
        'cosine_correlation': r_cos_avg,
        'cosine_p_value': p_cos_avg,
        'cosine_significant': p_cos_avg < 0.05,
        'euclidean_correlation': r_euc_avg,
        'euclidean_p_value': p_euc_avg,
        'euclidean_significant': p_euc_avg < 0.05
    })
    
    return pd.DataFrame(results)


def detect_outliers_isolation_forest(df, contamination=0.1):
    """Detect outliers using Isolation Forest."""
    embeddings = np.array(df['story_embedding'].tolist())
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = iso_forest.fit_predict(embeddings)
    
    # -1 = outlier, 1 = inlier
    df = df.copy()
    df['isolation_forest_outlier'] = outlier_labels == -1
    df['isolation_forest_score'] = iso_forest.score_samples(embeddings)
    
    return df


def detect_outliers_lof(df, n_neighbors=20, contamination=0.1):
    """Detect outliers using Local Outlier Factor."""
    embeddings = np.array(df['story_embedding'].tolist())
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    outlier_labels = lof.fit_predict(embeddings)
    
    df = df.copy()
    df['lof_outlier'] = outlier_labels == -1
    df['lof_score'] = lof.negative_outlier_factor_
    
    return df


def analyze_outlier_quality(df):
    """Analyze if outliers have lower quality scores."""
    results = []
    
    for outlier_type in ['isolation_forest_outlier', 'lof_outlier']:
        outliers = df[df[outlier_type]]
        inliers = df[~df[outlier_type]]
        
        for metric in TARGET_METRICS + ['average_score']:
            short_name = METRIC_SHORT_NAMES.get(metric, 'average_score')
            
            outlier_mean = outliers[metric].mean()
            inlier_mean = inliers[metric].mean()
            
            # t-test
            t_stat, p_value = stats.ttest_ind(outliers[metric], inliers[metric])
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(outliers)-1)*outliers[metric].std()**2 + (len(inliers)-1)*inliers[metric].std()**2) 
                / (len(outliers) + len(inliers) - 2)
            )
            cohens_d = (inlier_mean - outlier_mean) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'outlier_method': outlier_type.replace('_outlier', ''),
                'metric': metric,
                'metric_short': short_name,
                'outlier_count': len(outliers),
                'inlier_count': len(inliers),
                'outlier_mean_score': outlier_mean,
                'inlier_mean_score': inlier_mean,
                'score_difference': inlier_mean - outlier_mean,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d
            })
    
    return pd.DataFrame(results)


def compute_distance_percentiles(df):
    """Compute score statistics by distance percentile."""
    df = df.copy()
    
    # Create distance quartiles
    df['distance_quartile'] = pd.qcut(df['cosine_distance_from_ideal'], q=4, labels=['Q1 (Closest)', 'Q2', 'Q3', 'Q4 (Farthest)'])
    
    # Compute statistics by quartile
    results = []
    for quartile in ['Q1 (Closest)', 'Q2', 'Q3', 'Q4 (Farthest)']:
        quartile_data = df[df['distance_quartile'] == quartile]
        
        row = {
            'distance_quartile': quartile,
            'n_stories': len(quartile_data),
            'mean_distance': quartile_data['cosine_distance_from_ideal'].mean()
        }
        
        for metric in TARGET_METRICS + ['average_score']:
            short_name = METRIC_SHORT_NAMES.get(metric, 'average_score')
            row[f'{short_name}_mean'] = quartile_data[metric].mean()
            row[f'{short_name}_std'] = quartile_data[metric].std()
        
        results.append(row)
    
    return pd.DataFrame(results), df


def create_visualizations(df, correlations, outlier_analysis, percentile_stats, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Scatter plot: Distance vs Average Score
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    scatter = ax.scatter(df['cosine_distance_from_ideal'], df['average_score'], 
                         alpha=0.5, c=df['average_score'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Cosine Distance from Ideal Centroid', fontsize=11)
    ax.set_ylabel('Average Quality Score', fontsize=11)
    ax.set_title('Distance from Ideal vs. Quality Score', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Score')
    
    # Add trend line
    z = np.polyfit(df['cosine_distance_from_ideal'], df['average_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['cosine_distance_from_ideal'].min(), df['cosine_distance_from_ideal'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={correlations[correlations["metric_short"]=="average_score"]["cosine_correlation"].values[0]:.3f})')
    ax.legend()
    
    ax = axes[1]
    scatter = ax.scatter(df['euclidean_distance_from_ideal'], df['average_score'], 
                         alpha=0.5, c=df['average_score'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Euclidean Distance from Ideal Centroid', fontsize=11)
    ax.set_ylabel('Average Quality Score', fontsize=11)
    ax.set_title('Euclidean Distance vs. Quality Score', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Score')
    
    plt.suptitle('RQ12: Distance from Ideal Centroid vs. Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_distance_vs_quality.png', bbox_inches='tight')
    plt.close()
    
    # 2. Correlation bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(correlations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, correlations['cosine_correlation'], width, label='Cosine Distance', color='steelblue')
    bars2 = ax.bar(x + width/2, correlations['euclidean_correlation'], width, label='Euclidean Distance', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(correlations['metric_short'], rotation=45, ha='right')
    ax.set_ylabel('Correlation with Distance', fontsize=11)
    ax.set_title('RQ12: Correlation Between Distance and Quality Scores', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_correlation_chart.png', bbox_inches='tight')
    plt.close()
    
    # 3. Outlier analysis: Box plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, metric in enumerate(TARGET_METRICS[:4]):
        ax = axes[idx // 2, idx % 2]
        short_name = METRIC_SHORT_NAMES[metric]
        
        # Create data for box plot
        plot_data = []
        for outlier_type in ['isolation_forest_outlier', 'lof_outlier']:
            outlier_name = outlier_type.replace('_outlier', '').replace('_', ' ').title()
            for label, mask in [('Inlier', ~df[outlier_type]), ('Outlier', df[outlier_type])]:
                for val in df[mask][metric]:
                    plot_data.append({'Method': outlier_name, 'Status': label, 'Score': val})
        
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x='Method', y='Score', hue='Status', palette={'Inlier': 'lightgreen', 'Outlier': 'salmon'}, ax=ax)
        ax.set_title(f'{short_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(title='')
    
    plt.suptitle('RQ12: Quality Scores by Outlier Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_outlier_quality_boxplots.png', bbox_inches='tight')
    plt.close()
    
    # 4. Distance quartile analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quartiles = percentile_stats['distance_quartile'].tolist()
    avg_scores = percentile_stats['average_score_mean'].tolist()
    stds = percentile_stats['average_score_std'].tolist()
    
    colors = ['green', 'yellowgreen', 'orange', 'red']
    bars = ax.bar(quartiles, avg_scores, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    ax.set_xlabel('Distance Quartile', fontsize=11)
    ax.set_ylabel('Average Quality Score', fontsize=11)
    ax.set_title('RQ12: Quality Score by Distance Quartile\n(Q1=Closest to Ideal, Q4=Farthest)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_quartile_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 5. PCA visualization with outliers highlighted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    embeddings = np.array(df['story_embedding'].tolist())
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    for idx, outlier_col in enumerate(['isolation_forest_outlier', 'lof_outlier']):
        ax = axes[idx]
        
        inliers = ~df[outlier_col]
        outliers = df[outlier_col]
        
        ax.scatter(embeddings_2d[inliers, 0], embeddings_2d[inliers, 1], 
                   c='lightblue', alpha=0.5, label='Inliers', edgecolors='blue', linewidth=0.5)
        ax.scatter(embeddings_2d[outliers, 0], embeddings_2d[outliers, 1], 
                   c='red', alpha=0.8, label='Outliers', edgecolors='darkred', linewidth=1, s=80)
        
        method_name = outlier_col.replace('_outlier', '').replace('_', ' ').title()
        ax.set_title(f'{method_name} Outliers', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
    
    plt.suptitle('RQ12: PCA Visualization with Detected Outliers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_pca_outliers.png', bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap: Outlier detection by model and culture
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, outlier_col in enumerate(['isolation_forest_outlier', 'lof_outlier']):
        ax = axes[idx]
        
        pivot = df.pivot_table(
            values=outlier_col,
            index='model',
            columns='culture',
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': '% Outliers'})
        
        method_name = outlier_col.replace('_outlier', '').replace('_', ' ').title()
        ax.set_title(f'{method_name}: Outlier % by Model × Culture', fontsize=11, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq12_outlier_heatmap.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, correlations, outlier_analysis, percentile_stats, n_ideal_stories):
    """Generate summary of key findings."""
    # Get correlation with average score
    avg_corr = correlations[correlations['metric_short'] == 'average_score'].iloc[0]
    
    # Get outlier analysis for average score
    iso_avg = outlier_analysis[(outlier_analysis['outlier_method'] == 'isolation_forest') & 
                               (outlier_analysis['metric_short'] == 'average_score')].iloc[0]
    lof_avg = outlier_analysis[(outlier_analysis['outlier_method'] == 'lof') & 
                               (outlier_analysis['metric_short'] == 'average_score')].iloc[0]
    
    # Q1 vs Q4 score difference
    q1_score = percentile_stats[percentile_stats['distance_quartile'] == 'Q1 (Closest)']['average_score_mean'].values[0]
    q4_score = percentile_stats[percentile_stats['distance_quartile'] == 'Q4 (Farthest)']['average_score_mean'].values[0]
    
    summary = {
        'research_question': 'RQ12: Identifying Hallucination in Vector Space',
        'main_question': 'Are low-quality stories outliers in embedding space?',
        'n_stories': len(df),
        'n_ideal_stories': n_ideal_stories,
        
        # Distance-quality correlation
        'cosine_distance_correlation': avg_corr['cosine_correlation'],
        'cosine_distance_p_value': avg_corr['cosine_p_value'],
        'cosine_correlation_significant': avg_corr['cosine_significant'],
        
        # Outlier analysis
        'isolation_forest_outlier_count': iso_avg['outlier_count'],
        'isolation_forest_outlier_mean_score': iso_avg['outlier_mean_score'],
        'isolation_forest_inlier_mean_score': iso_avg['inlier_mean_score'],
        'isolation_forest_significant': iso_avg['significant'],
        
        'lof_outlier_count': lof_avg['outlier_count'],
        'lof_outlier_mean_score': lof_avg['outlier_mean_score'],
        'lof_inlier_mean_score': lof_avg['inlier_mean_score'],
        'lof_significant': lof_avg['significant'],
        
        # Quartile analysis
        'q1_closest_mean_score': q1_score,
        'q4_farthest_mean_score': q4_score,
        'q1_q4_difference': q1_score - q4_score,
        
        # Interpretation
        'distance_predicts_quality': avg_corr['cosine_correlation'] < -0.1 and avg_corr['cosine_significant'],
        'outliers_are_lower_quality': iso_avg['significant'] and lof_avg['significant']
    }
    
    return summary


def main():
    """Main execution function for RQ12 analysis."""
    print("=" * 70)
    print("RQ12: Identifying Hallucination in Vector Space")
    print("=" * 70)
    print("\nQuestion: Are low-quality stories outliers in embedding space?")
    print("Method: Distance from ideal centroid + outlier detection")
    print("-" * 70)
    
    if not HAS_SKLEARN:
        print("\nERROR: scikit-learn required. Install with: pip install scikit-learn")
        return
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading and merging data...")
    df = load_data()
    df = compute_average_score(df)
    print(f"    Loaded {len(df)} stories")
    print(f"    Score range: {df['average_score'].min():.2f} - {df['average_score'].max():.2f}")
    
    # Define ideal centroid
    print("\n[2] Defining ideal centroid (high-quality stories)...")
    ideal_centroid, n_ideal = define_ideal_centroid(df, threshold=4.0)
    print(f"    Ideal centroid computed from {n_ideal} high-quality stories")
    
    # Compute distances
    print("\n[3] Computing distances from ideal centroid...")
    df = compute_distances_from_ideal(df, ideal_centroid)
    print(f"    Cosine distance range: {df['cosine_distance_from_ideal'].min():.4f} - {df['cosine_distance_from_ideal'].max():.4f}")
    
    # Correlate with scores
    print("\n[4] Correlating distances with quality scores...")
    correlations = correlate_distance_with_scores(df)
    correlations.to_csv(output_dir / 'rq12_distance_correlations.csv', index=False)
    print(f"    Exported: rq12_distance_correlations.csv")
    
    print("\n    Distance-Quality Correlations:")
    for _, row in correlations.iterrows():
        sig = "*" if row['cosine_significant'] else ""
        print(f"      {row['metric_short']}: r = {row['cosine_correlation']:.4f} {sig}")
    
    # Outlier detection
    print("\n[5] Detecting outliers...")
    df = detect_outliers_isolation_forest(df, contamination=0.1)
    n_iso_outliers = df['isolation_forest_outlier'].sum()
    print(f"    Isolation Forest: {n_iso_outliers} outliers ({n_iso_outliers/len(df)*100:.1f}%)")
    
    df = detect_outliers_lof(df, contamination=0.1)
    n_lof_outliers = df['lof_outlier'].sum()
    print(f"    Local Outlier Factor: {n_lof_outliers} outliers ({n_lof_outliers/len(df)*100:.1f}%)")
    
    # Analyze outlier quality
    print("\n[6] Analyzing outlier quality scores...")
    outlier_analysis = analyze_outlier_quality(df)
    outlier_analysis.to_csv(output_dir / 'rq12_outlier_analysis.csv', index=False)
    print(f"    Exported: rq12_outlier_analysis.csv")
    
    # Print key findings
    print("\n    Outlier vs Inlier Quality Comparison:")
    for method in ['isolation_forest', 'lof']:
        avg_row = outlier_analysis[(outlier_analysis['outlier_method'] == method) & 
                                   (outlier_analysis['metric_short'] == 'average_score')].iloc[0]
        sig = "*" if avg_row['significant'] else ""
        print(f"      {method}: Outlier={avg_row['outlier_mean_score']:.3f}, Inlier={avg_row['inlier_mean_score']:.3f}, diff={avg_row['score_difference']:.3f} {sig}")
    
    # Distance percentile analysis
    print("\n[7] Computing distance percentile analysis...")
    percentile_stats, df = compute_distance_percentiles(df)
    percentile_stats.to_csv(output_dir / 'rq12_distance_percentiles.csv', index=False)
    print(f"    Exported: rq12_distance_percentiles.csv")
    
    print("\n    Quality by Distance Quartile:")
    for _, row in percentile_stats.iterrows():
        print(f"      {row['distance_quartile']}: avg_score = {row['average_score_mean']:.3f}")
    
    # Export detailed data
    print("\n[8] Exporting detailed story data...")
    export_cols = ['model', 'culture', 'story_type', 'region', 'average_score',
                   'cosine_distance_from_ideal', 'euclidean_distance_from_ideal',
                   'isolation_forest_outlier', 'isolation_forest_score',
                   'lof_outlier', 'lof_score', 'distance_quartile'] + TARGET_METRICS
    df[export_cols].to_csv(output_dir / 'rq12_story_analysis.csv', index=False)
    print(f"    Exported: rq12_story_analysis.csv")
    
    # Create visualizations
    print("\n[9] Creating visualizations...")
    create_visualizations(df, correlations, outlier_analysis, percentile_stats, output_dir)
    
    # Generate summary
    print("\n[10] Generating summary...")
    summary = generate_summary(df, correlations, outlier_analysis, percentile_stats, n_ideal)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq12_summary.csv', index=False)
    print(f"    Exported: rq12_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ12 ANALYSIS COMPLETE")
    print("=" * 70)
    
    avg_corr = correlations[correlations['metric_short'] == 'average_score'].iloc[0]
    
    print(f"\nKey Findings:")
    
    print(f"\n  Distance-Quality Correlation:")
    if avg_corr['cosine_correlation'] < -0.1 and avg_corr['cosine_significant']:
        print(f"    ✓ NEGATIVE correlation (r = {avg_corr['cosine_correlation']:.4f}, p < 0.05)")
        print(f"    → Stories far from ideal centroid tend to have LOWER quality")
    elif avg_corr['cosine_correlation'] < 0:
        print(f"    ⚠️  Weak negative correlation (r = {avg_corr['cosine_correlation']:.4f})")
    else:
        print(f"    ✗ No meaningful correlation (r = {avg_corr['cosine_correlation']:.4f})")
    
    iso_avg = outlier_analysis[(outlier_analysis['outlier_method'] == 'isolation_forest') & 
                               (outlier_analysis['metric_short'] == 'average_score')].iloc[0]
    lof_avg = outlier_analysis[(outlier_analysis['outlier_method'] == 'lof') & 
                               (outlier_analysis['metric_short'] == 'average_score')].iloc[0]
    
    print(f"\n  Outlier Analysis:")
    print(f"    Isolation Forest: {iso_avg['outlier_count']} outliers detected")
    if iso_avg['significant']:
        print(f"      ✓ Outliers have LOWER scores ({iso_avg['outlier_mean_score']:.3f} vs {iso_avg['inlier_mean_score']:.3f})")
    else:
        print(f"      ✗ No significant difference")
    
    print(f"    LOF: {lof_avg['outlier_count']} outliers detected")
    if lof_avg['significant']:
        print(f"      ✓ Outliers have LOWER scores ({lof_avg['outlier_mean_score']:.3f} vs {lof_avg['inlier_mean_score']:.3f})")
    else:
        print(f"      ✗ No significant difference")
    
    q1 = percentile_stats[percentile_stats['distance_quartile'] == 'Q1 (Closest)']['average_score_mean'].values[0]
    q4 = percentile_stats[percentile_stats['distance_quartile'] == 'Q4 (Farthest)']['average_score_mean'].values[0]
    
    print(f"\n  Quartile Analysis:")
    print(f"    Q1 (closest to ideal): {q1:.3f}")
    print(f"    Q4 (farthest from ideal): {q4:.3f}")
    print(f"    Difference: {q1-q4:.3f}")
    
    # Overall conclusion
    if summary['distance_predicts_quality'] or summary['outliers_are_lower_quality']:
        print(f"\n  ✓ CONCLUSION: Embedding distance CAN help identify low-quality stories")
        print(f"    Stories distant from ideal centroid or detected as outliers")
        print(f"    tend to have lower human evaluation scores.")
    else:
        print(f"\n  ⚠️  CONCLUSION: Limited ability to detect hallucination via embeddings")
        print(f"    Distance from ideal centroid is not a strong quality predictor.")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq12_distance_correlations.csv - Distance-score correlations")
    print("  2. rq12_outlier_analysis.csv - Outlier vs inlier comparison")
    print("  3. rq12_distance_percentiles.csv - Quality by distance quartile")
    print("  4. rq12_story_analysis.csv - Detailed per-story analysis")
    print("  5. rq12_summary.csv - Key findings")
    print("-" * 70)
    
    return df, correlations, outlier_analysis


if __name__ == "__main__":
    df, correlations, outlier_analysis = main()
