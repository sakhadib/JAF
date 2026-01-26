"""
RQ2: Bias in Representation of Low-Resource Cultures
=====================================================
Question: Is there a significant performance disparity in generated story quality 
          between dominant cultures (Bengali) and marginalized indigenous groups?

Method:
- Create binary grouping: Bengali vs. Others (11 cultures aggregated)
- Compare means using Independent samples t-test
- Per-culture analysis: Group by all 12 cultures, rank by each metric
- Visualize: Heatmap of culture × metric
"""

import pandas as pd
import numpy as np
from scipy import stats
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


def load_data():
    """Load the evaluation dataset."""
    data_path = Path(__file__).parent.parent / "data" / "evaluated_stories.csv"
    df = pd.read_csv(data_path)
    return df


def get_evaluation_metrics():
    """Return the list of evaluation metric columns."""
    return [
        'Cultural Accuracy & Authenticity',
        'Contextual & Temporal Appropriateness',
        'Narrative & Symbolic Coherence',
        'Linguistic & Expressive Appropriateness (Bangla)'
    ]


def create_culture_grouping(df):
    """Create binary grouping: Bengali (dominant) vs. Indigenous (others)."""
    df = df.copy()
    df['culture_group'] = df['culture'].apply(
        lambda x: 'Bengali (Dominant)' if x == 'Bengali' else 'Indigenous (11 groups)'
    )
    return df


def calculate_culture_stats(df, metrics):
    """Calculate descriptive statistics for each culture across all metrics."""
    results = []
    
    for culture in df['culture'].unique():
        culture_data = df[df['culture'] == culture]
        row = {'culture': culture, 'n': len(culture_data)}
        
        for metric in metrics:
            row[f'{metric}_mean'] = culture_data[metric].mean()
            row[f'{metric}_std'] = culture_data[metric].std()
            row[f'{metric}_median'] = culture_data[metric].median()
        
        # Calculate overall average across all metrics
        row['overall_mean'] = np.mean([row[f'{m}_mean'] for m in metrics])
        results.append(row)
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values('overall_mean', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def perform_binary_comparison(df, metrics):
    """Perform t-test comparing Bengali vs. Indigenous groups."""
    df_grouped = create_culture_grouping(df)
    
    bengali = df_grouped[df_grouped['culture_group'] == 'Bengali (Dominant)']
    indigenous = df_grouped[df_grouped['culture_group'] == 'Indigenous (11 groups)']
    
    results = []
    
    for metric in metrics:
        bengali_scores = bengali[metric]
        indigenous_scores = indigenous[metric]
        
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(bengali_scores, indigenous_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(bengali_scores) - 1) * bengali_scores.std()**2 + 
             (len(indigenous_scores) - 1) * indigenous_scores.std()**2) / 
            (len(bengali_scores) + len(indigenous_scores) - 2)
        )
        cohens_d = (bengali_scores.mean() - indigenous_scores.mean()) / pooled_std
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "Small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        
        # Significance interpretation
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # Direction of bias
        if bengali_scores.mean() > indigenous_scores.mean():
            bias_direction = "Favors Bengali"
        elif bengali_scores.mean() < indigenous_scores.mean():
            bias_direction = "Favors Indigenous"
        else:
            bias_direction = "No difference"
        
        results.append({
            'metric': metric,
            'bengali_n': len(bengali_scores),
            'bengali_mean': round(bengali_scores.mean(), 4),
            'bengali_std': round(bengali_scores.std(), 4),
            'indigenous_n': len(indigenous_scores),
            'indigenous_mean': round(indigenous_scores.mean(), 4),
            'indigenous_std': round(indigenous_scores.std(), 4),
            'mean_difference': round(bengali_scores.mean() - indigenous_scores.mean(), 4),
            't_statistic': round(t_stat, 4),
            'p_value': p_value,
            'p_value_formatted': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
            'significance': significance,
            'cohens_d': round(cohens_d, 4),
            'effect_size': effect_interpretation,
            'bias_direction': bias_direction
        })
    
    return pd.DataFrame(results)


def perform_anova_per_metric(df, metrics):
    """Perform one-way ANOVA for each metric across all 12 cultures."""
    results = []
    
    for metric in metrics:
        groups = [group[metric].values for name, group in df.groupby('culture')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        grand_mean = df[metric].mean()
        ss_total = ((df[metric] - grand_mean) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        eta_squared = ss_between / ss_total
        
        results.append({
            'metric': metric,
            'f_statistic': round(f_stat, 4),
            'p_value': p_value,
            'p_value_formatted': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
            'significant': p_value < 0.05,
            'eta_squared': round(eta_squared, 4)
        })
    
    return pd.DataFrame(results)


def create_visualizations(df, metrics, culture_stats, binary_comparison, output_dir):
    """Create and save visualizations."""
    if not HAS_PLOTTING:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Heatmap: Culture × Metric
    print("  Creating heatmap...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data for heatmap
    heatmap_data = []
    for culture in culture_stats['culture']:
        row = []
        for metric in metrics:
            mean_val = df[df['culture'] == culture][metric].mean()
            row.append(mean_val)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=culture_stats['culture'],
        columns=[m.replace(' & ', '\n& ').replace('(Bangla)', '\n(Bangla)') for m in metrics]
    )
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=2.5,
        vmin=0,
        vmax=5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Mean Score (0-5)'}
    )
    
    ax.set_title('RQ2: Evaluation Scores by Culture\n(Higher is Better)', fontsize=14)
    ax.set_xlabel('Evaluation Metric', fontsize=12)
    ax.set_ylabel('Culture', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq2_heatmap_culture_metric.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot: Bengali vs Indigenous comparison
    print("  Creating binary comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    df_grouped = create_culture_grouping(df)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group means
        means = df_grouped.groupby('culture_group')[metric].mean()
        stds = df_grouped.groupby('culture_group')[metric].std()
        counts = df_grouped.groupby('culture_group')[metric].count()
        cis = 1.96 * stds / np.sqrt(counts)
        
        groups = ['Bengali (Dominant)', 'Indigenous (11 groups)']
        colors = ['#e74c3c', '#3498db']
        
        bars = ax.bar(groups, [means[g] for g in groups], 
                     yerr=[cis[g] for g in groups],
                     capsize=5, color=colors, edgecolor='black', linewidth=1)
        
        # Get p-value from binary comparison
        p_val = binary_comparison[binary_comparison['metric'] == metric]['p_value'].values[0]
        sig = binary_comparison[binary_comparison['metric'] == metric]['significance'].values[0]
        
        ax.set_title(f'{metric}\n(p={p_val:.4f} {sig})', fontsize=10)
        ax.set_ylabel('Mean Score', fontsize=10)
        ax.set_ylim(0, 5)
        
        # Add value labels
        for bar, group in zip(bars, groups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + cis[group] + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('RQ2: Bengali vs. Indigenous Groups Comparison\n(Error bars: 95% CI)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'rq2_binary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot: All cultures for Cultural Accuracy metric
    print("  Creating culture ranking plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Order by mean score
    order = culture_stats['culture'].tolist()
    
    # Color Bengali differently
    palette = ['#e74c3c' if c == 'Bengali' else '#3498db' for c in order]
    
    sns.boxplot(
        data=df,
        x='culture',
        y='Cultural Accuracy & Authenticity',
        order=order,
        palette=palette,
        ax=ax
    )
    
    ax.set_xlabel('Culture', fontsize=12)
    ax.set_ylabel('Cultural Accuracy & Authenticity Score', fontsize=12)
    ax.set_title('RQ2: Cultural Accuracy Scores by Culture\n(Bengali in Red, Indigenous in Blue)', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Bengali (Dominant)'),
        Patch(facecolor='#3498db', label='Indigenous Groups')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq2_culture_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Radar/Spider chart for overall culture comparison
    print("  Creating radar chart...")
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Prepare data
    categories = [m.split('&')[0].strip() for m in metrics]  # Shortened names
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the circle
    
    # Plot Bengali
    bengali_values = [df[df['culture'] == 'Bengali'][m].mean() for m in metrics]
    bengali_values += bengali_values[:1]
    ax.plot(angles, bengali_values, 'o-', linewidth=2, label='Bengali', color='#e74c3c')
    ax.fill(angles, bengali_values, alpha=0.25, color='#e74c3c')
    
    # Plot Indigenous average
    indigenous_values = [df[df['culture'] != 'Bengali'][m].mean() for m in metrics]
    indigenous_values += indigenous_values[:1]
    ax.plot(angles, indigenous_values, 'o-', linewidth=2, label='Indigenous (avg)', color='#3498db')
    ax.fill(angles, indigenous_values, alpha=0.25, color='#3498db')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_title('RQ2: Bengali vs. Indigenous Average Scores\n(Radar Chart)', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq2_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualizations to {output_dir}")


def run_rq2_analysis():
    """Main function to run RQ2 analysis."""
    print("=" * 60)
    print("RQ2: Bias in Representation of Low-Resource Cultures")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    result_dir = base_dir / "result"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    metrics = get_evaluation_metrics()
    print(f"  ✓ Loaded {len(df)} records")
    print(f"  ✓ Cultures: {df['culture'].nunique()} unique")
    print(f"  ✓ Bengali: {len(df[df['culture'] == 'Bengali'])} stories")
    print(f"  ✓ Indigenous: {len(df[df['culture'] != 'Bengali'])} stories")
    
    # Per-culture statistics
    print("\n[2/6] Calculating per-culture statistics...")
    culture_stats = calculate_culture_stats(df, metrics)
    culture_stats.to_csv(result_dir / 'rq2_culture_stats.csv', index=False)
    print(f"  ✓ Saved: rq2_culture_stats.csv")
    
    # Print culture ranking
    print("\n  Culture Rankings (by overall mean score):")
    print("  " + "-" * 55)
    for _, row in culture_stats.iterrows():
        culture_type = "Bengali" if row['culture'] == 'Bengali' else "Indigenous"
        print(f"  #{int(row['rank']):2d}: {row['culture']:<20} Overall={row['overall_mean']:.3f} ({culture_type})")
    
    # Binary comparison: Bengali vs Indigenous
    print("\n[3/6] Performing Bengali vs. Indigenous comparison...")
    binary_comparison = perform_binary_comparison(df, metrics)
    binary_comparison.to_csv(result_dir / 'rq2_binary_comparison.csv', index=False)
    print(f"  ✓ Saved: rq2_binary_comparison.csv")
    
    print("\n  Binary Comparison Results (Bengali vs. Indigenous):")
    print("  " + "-" * 70)
    for _, row in binary_comparison.iterrows():
        metric_short = row['metric'].split('&')[0].strip()[:25]
        print(f"  {metric_short:<25} Bengali={row['bengali_mean']:.2f} vs Indigenous={row['indigenous_mean']:.2f} "
              f"Diff={row['mean_difference']:+.2f} p={row['p_value_formatted']} {row['significance']}")
    
    # ANOVA across all 12 cultures
    print("\n[4/6] Performing ANOVA across all 12 cultures...")
    anova_results = perform_anova_per_metric(df, metrics)
    anova_results.to_csv(result_dir / 'rq2_anova_cultures.csv', index=False)
    print(f"  ✓ Saved: rq2_anova_cultures.csv")
    
    print("\n  ANOVA Results (12 cultures):")
    print("  " + "-" * 55)
    for _, row in anova_results.iterrows():
        metric_short = row['metric'].split('&')[0].strip()[:30]
        sig = "✓" if row['significant'] else "✗"
        print(f"  {metric_short:<30} F={row['f_statistic']:.2f}, p={row['p_value_formatted']} [{sig}]")
    
    # Visualizations
    print("\n[5/6] Creating visualizations...")
    create_visualizations(df, metrics, culture_stats, binary_comparison, result_dir)
    
    # Summary
    print("\n[6/6] Generating summary...")
    
    # Determine overall bias
    sig_metrics = binary_comparison[binary_comparison['significance'] != 'ns']
    if len(sig_metrics) == 0:
        overall_bias = "No significant bias detected"
    else:
        favors_bengali = (sig_metrics['bias_direction'] == 'Favors Bengali').sum()
        favors_indigenous = (sig_metrics['bias_direction'] == 'Favors Indigenous').sum()
        if favors_bengali > favors_indigenous:
            overall_bias = f"Bias favoring Bengali ({favors_bengali}/{len(sig_metrics)} significant metrics)"
        elif favors_indigenous > favors_bengali:
            overall_bias = f"Bias favoring Indigenous ({favors_indigenous}/{len(sig_metrics)} significant metrics)"
        else:
            overall_bias = "Mixed bias pattern"
    
    # Find best and worst performing cultures
    best_culture = culture_stats.iloc[0]['culture']
    worst_culture = culture_stats.iloc[-1]['culture']
    
    summary = {
        'research_question': 'RQ2',
        'question': 'Is there performance disparity between Bengali and indigenous cultures?',
        'n_total': len(df),
        'n_bengali': len(df[df['culture'] == 'Bengali']),
        'n_indigenous': len(df[df['culture'] != 'Bengali']),
        'n_cultures': df['culture'].nunique(),
        'bengali_overall_mean': round(df[df['culture'] == 'Bengali'][metrics].mean().mean(), 4),
        'indigenous_overall_mean': round(df[df['culture'] != 'Bengali'][metrics].mean().mean(), 4),
        'significant_differences_count': len(sig_metrics),
        'total_metrics_tested': len(metrics),
        'overall_bias_finding': overall_bias,
        'best_culture': best_culture,
        'best_culture_mean': round(culture_stats.iloc[0]['overall_mean'], 4),
        'worst_culture': worst_culture,
        'worst_culture_mean': round(culture_stats.iloc[-1]['overall_mean'], 4),
        'performance_gap': round(culture_stats.iloc[0]['overall_mean'] - culture_stats.iloc[-1]['overall_mean'], 4)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(result_dir / 'rq2_summary.csv', index=False)
    print(f"  ✓ Saved: rq2_summary.csv")
    
    # Create detailed metric-by-culture matrix for export
    metric_matrix = df.groupby('culture')[metrics].mean().round(4)
    metric_matrix['overall_mean'] = metric_matrix.mean(axis=1).round(4)
    metric_matrix = metric_matrix.sort_values('overall_mean', ascending=False)
    metric_matrix.to_csv(result_dir / 'rq2_metric_culture_matrix.csv')
    print(f"  ✓ Saved: rq2_metric_culture_matrix.csv")
    
    print("\n" + "=" * 60)
    print("RQ2 Analysis Complete!")
    print("=" * 60)
    print(f"\n  Key Finding: {overall_bias}")
    print(f"  Bengali overall mean: {summary['bengali_overall_mean']:.3f}")
    print(f"  Indigenous overall mean: {summary['indigenous_overall_mean']:.3f}")
    print(f"  Best culture: {best_culture} ({culture_stats.iloc[0]['overall_mean']:.3f})")
    print(f"  Worst culture: {worst_culture} ({culture_stats.iloc[-1]['overall_mean']:.3f})")
    
    print(f"\nOutput files in: {result_dir}")
    print("  - rq2_culture_stats.csv")
    print("  - rq2_binary_comparison.csv")
    print("  - rq2_anova_cultures.csv")
    print("  - rq2_metric_culture_matrix.csv")
    print("  - rq2_summary.csv")
    if HAS_PLOTTING:
        print("  - rq2_heatmap_culture_metric.png")
        print("  - rq2_binary_comparison.png")
        print("  - rq2_culture_boxplot.png")
        print("  - rq2_radar_chart.png")
    
    return {
        'culture_stats': culture_stats,
        'binary_comparison': binary_comparison,
        'anova_results': anova_results,
        'summary': summary
    }


if __name__ == "__main__":
    results = run_rq2_analysis()
