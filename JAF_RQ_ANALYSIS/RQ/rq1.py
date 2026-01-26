"""
RQ1: Comparative Model Efficacy in Cultural Authenticity
=========================================================
Question: Which LLM architecture yields the highest human-rated performance 
          for preservation of indigenous cultural norms?

Method:
- Group by `model`
- Calculate descriptive statistics (Mean, Std, Median)
- One-way ANOVA test for significance
- Tukey HSD post-hoc test for pairwise comparisons
- Visualization: Box plot / Violin plot
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization and advanced stats libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Skipping Tukey HSD test.")


def load_data():
    """Load the evaluation dataset."""
    data_path = Path(__file__).parent.parent / "data" / "evaluated_stories.csv"
    df = pd.read_csv(data_path)
    return df


def calculate_descriptive_stats(df, metric_col, group_col='model'):
    """Calculate descriptive statistics grouped by model."""
    stats_df = df.groupby(group_col)[metric_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(4)
    
    # Add confidence interval (95%)
    stats_df['ci_95'] = (1.96 * stats_df['std'] / np.sqrt(stats_df['count'])).round(4)
    stats_df['ci_lower'] = (stats_df['mean'] - stats_df['ci_95']).round(4)
    stats_df['ci_upper'] = (stats_df['mean'] + stats_df['ci_95']).round(4)
    
    # Rank by mean (descending)
    stats_df['rank'] = stats_df['mean'].rank(ascending=False).astype(int)
    stats_df = stats_df.sort_values('rank')
    
    return stats_df


def perform_anova(df, metric_col, group_col='model'):
    """Perform one-way ANOVA test."""
    groups = [group[metric_col].values for name, group in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    # eta^2 = SS_between / SS_total
    grand_mean = df[metric_col].mean()
    ss_total = ((df[metric_col] - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_squared = ss_between / ss_total
    
    # Interpretation
    if p_value < 0.001:
        significance = "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not significant (p >= 0.05)"
    
    # Effect size interpretation
    if eta_squared < 0.01:
        effect_interpretation = "Negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "Small"
    elif eta_squared < 0.14:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"
    
    anova_results = {
        'test': 'One-way ANOVA',
        'f_statistic': round(f_stat, 4),
        'p_value': p_value,
        'p_value_formatted': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        'significance': significance,
        'eta_squared': round(eta_squared, 4),
        'effect_size_interpretation': effect_interpretation,
        'degrees_of_freedom_between': len(groups) - 1,
        'degrees_of_freedom_within': len(df) - len(groups),
        'n_groups': len(groups),
        'n_total': len(df)
    }
    
    return anova_results


def perform_tukey_hsd(df, metric_col, group_col='model'):
    """Perform Tukey HSD post-hoc test for pairwise comparisons."""
    if not HAS_STATSMODELS:
        return None
    
    tukey = pairwise_tukeyhsd(
        endog=df[metric_col],
        groups=df[group_col],
        alpha=0.05
    )
    
    # Convert to DataFrame using the summary data
    # The summary_frame method provides a clean DataFrame
    try:
        tukey_df = tukey.summary().as_html()
        # Parse from the _results_table attribute
        results_data = []
        for i in range(len(tukey._results_table.data) - 1):
            row = tukey._results_table.data[i + 1]  # Skip header
            results_data.append(row)
        
        tukey_df = pd.DataFrame(results_data, columns=['group1', 'group2', 'mean_diff', 'p_adj', 'lower_ci', 'upper_ci', 'reject_null'])
        
        # Convert types
        tukey_df['mean_diff'] = pd.to_numeric(tukey_df['mean_diff']).round(4)
        tukey_df['p_adj'] = pd.to_numeric(tukey_df['p_adj']).round(4)
        tukey_df['lower_ci'] = pd.to_numeric(tukey_df['lower_ci']).round(4)
        tukey_df['upper_ci'] = pd.to_numeric(tukey_df['upper_ci']).round(4)
        tukey_df['reject_null'] = tukey_df['reject_null'].astype(bool)
        
    except Exception:
        # Fallback: manually extract from tukey object attributes
        n_groups = len(tukey.groupsunique)
        pairs = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                pairs.append((i, j))
        
        tukey_df = pd.DataFrame({
            'group1': [tukey.groupsunique[p[0]] for p in pairs],
            'group2': [tukey.groupsunique[p[1]] for p in pairs],
            'mean_diff': tukey.meandiffs.round(4),
            'p_adj': tukey.pvalues.round(4),
            'lower_ci': tukey.confint[:, 0].round(4),
            'upper_ci': tukey.confint[:, 1].round(4),
            'reject_null': tukey.reject
        })
    
    # Add significance interpretation
    tukey_df['significance'] = tukey_df['p_adj'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    
    return tukey_df


def create_visualizations(df, metric_col, output_dir, group_col='model'):
    """Create and save visualizations."""
    if not HAS_PLOTTING:
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Box Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Order by mean score
    order = df.groupby(group_col)[metric_col].mean().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df, 
        x=group_col, 
        y=metric_col, 
        order=order,
        palette='viridis',
        ax=ax
    )
    
    # Add mean markers
    means = df.groupby(group_col)[metric_col].mean()
    for i, model in enumerate(order):
        ax.scatter(i, means[model], color='red', s=100, zorder=5, marker='D', label='Mean' if i == 0 else '')
    
    ax.set_xlabel('Generation Model', fontsize=12)
    ax.set_ylabel('Cultural Accuracy & Authenticity Score', fontsize=12)
    ax.set_title('RQ1: Model Comparison - Cultural Accuracy & Authenticity\n(Higher is Better, Scale: 0-5)', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq1_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Violin Plot with individual points
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(
        data=df, 
        x=group_col, 
        y=metric_col, 
        order=order,
        palette='viridis',
        inner=None,
        ax=ax
    )
    
    # Add strip plot for individual points
    sns.stripplot(
        data=df, 
        x=group_col, 
        y=metric_col, 
        order=order,
        color='black',
        alpha=0.3,
        size=3,
        ax=ax
    )
    
    ax.set_xlabel('Generation Model', fontsize=12)
    ax.set_ylabel('Cultural Accuracy & Authenticity Score', fontsize=12)
    ax.set_title('RQ1: Score Distribution by Model (Violin Plot)', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq1_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot with error bars (confidence intervals)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats_df = df.groupby(group_col)[metric_col].agg(['mean', 'std', 'count'])
    stats_df['ci'] = 1.96 * stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df = stats_df.loc[order]
    
    bars = ax.bar(
        range(len(order)), 
        stats_df['mean'], 
        yerr=stats_df['ci'],
        capsize=5,
        color=sns.color_palette('viridis', len(order)),
        edgecolor='black',
        linewidth=1
    )
    
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_xlabel('Generation Model', fontsize=12)
    ax.set_ylabel('Mean Cultural Accuracy Score', fontsize=12)
    ax.set_title('RQ1: Mean Scores with 95% Confidence Intervals', fontsize=14)
    ax.set_ylim(0, 5)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, stats_df['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stats_df['ci'].iloc[i] + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq1_barplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualizations to {output_dir}")


def run_rq1_analysis():
    """Main function to run RQ1 analysis."""
    print("=" * 60)
    print("RQ1: Comparative Model Efficacy in Cultural Authenticity")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    result_dir = base_dir / "result"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    metric_col = 'Cultural Accuracy & Authenticity'
    print(f"  ✓ Loaded {len(df)} records")
    print(f"  ✓ Models: {df['model'].nunique()} unique")
    
    # Descriptive statistics
    print("\n[2/5] Calculating descriptive statistics...")
    stats_df = calculate_descriptive_stats(df, metric_col)
    stats_df.to_csv(result_dir / 'rq1_descriptive_stats.csv')
    print(f"  ✓ Saved: rq1_descriptive_stats.csv")
    
    # Print summary
    print("\n  Model Rankings (by mean Cultural Accuracy):")
    print("  " + "-" * 50)
    for idx, row in stats_df.iterrows():
        print(f"  #{int(row['rank'])}: {idx:<32} Mean={row['mean']:.3f} (±{row['std']:.3f})")
    
    # ANOVA test
    print("\n[3/5] Performing One-way ANOVA...")
    anova_results = perform_anova(df, metric_col)
    anova_df = pd.DataFrame([anova_results])
    anova_df.to_csv(result_dir / 'rq1_anova_results.csv', index=False)
    print(f"  ✓ Saved: rq1_anova_results.csv")
    print(f"\n  ANOVA Results:")
    print(f"  " + "-" * 50)
    print(f"  F-statistic: {anova_results['f_statistic']}")
    print(f"  p-value: {anova_results['p_value_formatted']}")
    print(f"  Result: {anova_results['significance']}")
    print(f"  Effect size (η²): {anova_results['eta_squared']} ({anova_results['effect_size_interpretation']})")
    
    # Tukey HSD post-hoc test
    print("\n[4/5] Performing Tukey HSD post-hoc test...")
    if HAS_STATSMODELS:
        tukey_df = perform_tukey_hsd(df, metric_col)
        tukey_df.to_csv(result_dir / 'rq1_tukey_hsd.csv', index=False)
        print(f"  ✓ Saved: rq1_tukey_hsd.csv")
        
        # Print significant pairwise differences
        sig_pairs = tukey_df[tukey_df['reject_null']]
        if len(sig_pairs) > 0:
            print(f"\n  Significant Pairwise Differences (p < 0.05):")
            print("  " + "-" * 50)
            for _, row in sig_pairs.iterrows():
                print(f"  {row['group1']} vs {row['group2']}: diff={row['mean_diff']:.3f}, p={row['p_adj']:.4f} {row['significance']}")
        else:
            print("  No significant pairwise differences found.")
    else:
        print("  ⚠ Skipped (statsmodels not available)")
    
    # Visualizations
    print("\n[5/5] Creating visualizations...")
    create_visualizations(df, metric_col, result_dir)
    
    # Summary export
    summary = {
        'research_question': 'RQ1',
        'question': 'Which LLM yields the highest human-rated cultural authenticity?',
        'metric_analyzed': metric_col,
        'n_total': len(df),
        'n_models': df['model'].nunique(),
        'best_model': stats_df.index[0],
        'best_model_mean': stats_df.iloc[0]['mean'],
        'worst_model': stats_df.index[-1],
        'worst_model_mean': stats_df.iloc[-1]['mean'],
        'anova_f_statistic': anova_results['f_statistic'],
        'anova_p_value': anova_results['p_value'],
        'anova_significant': anova_results['p_value'] < 0.05,
        'effect_size_eta_squared': anova_results['eta_squared'],
        'effect_size_interpretation': anova_results['effect_size_interpretation']
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(result_dir / 'rq1_summary.csv', index=False)
    print(f"  ✓ Saved: rq1_summary.csv")
    
    print("\n" + "=" * 60)
    print("RQ1 Analysis Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {result_dir}")
    print("  - rq1_descriptive_stats.csv")
    print("  - rq1_anova_results.csv")
    print("  - rq1_tukey_hsd.csv")
    print("  - rq1_summary.csv")
    if HAS_PLOTTING:
        print("  - rq1_boxplot.png")
        print("  - rq1_violin.png")
        print("  - rq1_barplot.png")
    
    return {
        'descriptive_stats': stats_df,
        'anova_results': anova_results,
        'tukey_hsd': tukey_df if HAS_STATSMODELS else None,
        'summary': summary
    }


if __name__ == "__main__":
    results = run_rq1_analysis()
