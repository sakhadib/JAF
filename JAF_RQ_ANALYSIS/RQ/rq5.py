"""
RQ5: Regional Contextual Performance
=====================================
Question: Is there better performance on prominent regions vs. remote areas?
          Do LLMs show regional bias in generating culturally accurate stories?

Method:
- Join CSV evaluations with parquet to get `region`
- Group by region, calculate mean scores per region
- Categorize regions: Prominent (Dhaka-adjacent) vs. Remote (Hill Tracts)
- Statistical test: ANOVA across regions + t-test for binary comparison

Required Join: CSV + Parquet on (model, culture, story)

Regions in dataset (9):
- Chittagong Hill Tracts (150) - Remote
- Sylhet (border areas) (100) - Remote
- Mymensingh (50) - Semi-Prominent
- Rajshahi, Rangpur, northwestern plains (50) - Prominent
- Cox's Bazar, coastal areas (50) - Remote
- Mymensingh, Sherpur (50) - Semi-Prominent
- Bandarban (50) - Remote (Hill Tracts)
- Bangladesh-Scoped (50) - General/Prominent
- North Bengal (49) - Semi-Prominent
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

# Try to import statsmodels for Tukey HSD
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Skipping Tukey HSD test.")


def load_csv_data():
    """Load the evaluation dataset from CSV."""
    data_path = Path(__file__).parent.parent / "data" / "evaluated_stories.csv"
    df = pd.read_csv(data_path)
    return df


def load_parquet_data():
    """Load parquet data to get region column."""
    parquet_dir = Path(__file__).parent.parent / "data" / "parquet"
    parquet_file = parquet_dir / "openai_text-embedding-3-large.parquet"
    
    if not parquet_file.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/parquet/")
        parquet_file = parquet_files[0]
    
    df = pd.read_parquet(parquet_file)
    return df[['model', 'culture', 'story', 'region']]


def merge_data(df_csv, df_parquet):
    """Merge CSV evaluations with parquet metadata to get region."""
    df_merged = pd.merge(
        df_csv,
        df_parquet[['model', 'culture', 'story', 'region']],
        on=['model', 'culture', 'story'],
        how='left'
    )
    
    missing = df_merged['region'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows could not be matched with parquet data")
    
    return df_merged


def get_evaluation_metrics():
    """Return the list of evaluation metric columns."""
    return [
        'Cultural Accuracy & Authenticity',
        'Contextual & Temporal Appropriateness',
        'Narrative & Symbolic Coherence',
        'Linguistic & Expressive Appropriateness (Bangla)'
    ]


def categorize_regions(df):
    """
    Categorize regions into Prominent vs Remote.
    
    Remote regions (Hill Tracts, border areas, coastal):
    - Chittagong Hill Tracts
    - Bandarban
    - Sylhet (border areas)
    - Cox's Bazar, coastal areas
    
    Prominent regions (central, well-documented):
    - Bangladesh-Scoped (general)
    - Rajshahi, Rangpur, northwestern plains
    - Mymensingh
    - Mymensingh, Sherpur
    - North Bengal
    """
    remote_regions = [
        'Chittagong Hill Tracts',
        'Bandarban',
        'Sylhet (border areas)',
        "Cox's Bazar, coastal areas"
    ]
    
    df = df.copy()
    df['region_category'] = df['region'].apply(
        lambda x: 'Remote' if x in remote_regions else 'Prominent'
    )
    
    return df


def calculate_region_stats(df, metrics):
    """Calculate descriptive statistics for each region across all metrics."""
    results = []
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        row = {
            'region': region,
            'n': len(region_data),
            'n_cultures': region_data['culture'].nunique(),
            'n_models': region_data['model'].nunique()
        }
        
        for metric in metrics:
            row[f'{metric}_mean'] = region_data[metric].mean()
            row[f'{metric}_std'] = region_data[metric].std()
            row[f'{metric}_median'] = region_data[metric].median()
        
        # Calculate overall average across all metrics
        row['overall_mean'] = np.mean([row[f'{m}_mean'] for m in metrics])
        results.append(row)
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values('overall_mean', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def perform_binary_comparison(df, metrics):
    """Perform t-test comparing Prominent vs Remote regions."""
    df_cat = categorize_regions(df)
    
    prominent = df_cat[df_cat['region_category'] == 'Prominent']
    remote = df_cat[df_cat['region_category'] == 'Remote']
    
    results = []
    
    for metric in metrics:
        prominent_scores = prominent[metric]
        remote_scores = remote[metric]
        
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(prominent_scores, remote_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(prominent_scores) - 1) * prominent_scores.std()**2 + 
             (len(remote_scores) - 1) * remote_scores.std()**2) / 
            (len(prominent_scores) + len(remote_scores) - 2)
        )
        cohens_d = (prominent_scores.mean() - remote_scores.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'metric': metric,
            'prominent_n': len(prominent_scores),
            'prominent_mean': prominent_scores.mean(),
            'prominent_std': prominent_scores.std(),
            'remote_n': len(remote_scores),
            'remote_mean': remote_scores.mean(),
            'remote_std': remote_scores.std(),
            'mean_difference': prominent_scores.mean() - remote_scores.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'favors': 'Prominent' if prominent_scores.mean() > remote_scores.mean() else 'Remote'
        })
    
    return pd.DataFrame(results)


def perform_anova_regions(df, metric):
    """Perform one-way ANOVA testing region effect on a metric."""
    regions = df['region'].unique()
    groups = [df[df['region'] == r][metric].dropna() for r in regions]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'f_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    grand_mean = df[metric].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df[metric] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < 0.05
    }


def perform_tukey_hsd(df, metric, group_col='region'):
    """Perform Tukey HSD post-hoc test for pairwise comparisons."""
    if not HAS_STATSMODELS:
        return None
    
    try:
        tukey = pairwise_tukeyhsd(
            endog=df[metric].dropna(),
            groups=df.loc[df[metric].notna(), group_col],
            alpha=0.05
        )
        
        results = []
        table = tukey._results_table
        headers = [str(h) for h in table[0]]
        
        for row in table[1:]:
            row_dict = {headers[i]: row[i] for i in range(len(headers))}
            results.append(row_dict)
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Tukey HSD failed: {e}")
        return None


def calculate_model_by_region_stats(df, metric):
    """Calculate statistics for each model × region combination."""
    results = []
    
    for model in df['model'].unique():
        for region in df['region'].unique():
            subset = df[(df['model'] == model) & (df['region'] == region)]
            if len(subset) > 0:
                results.append({
                    'model': model,
                    'region': region,
                    'n': len(subset),
                    'mean': subset[metric].mean(),
                    'std': subset[metric].std(),
                    'median': subset[metric].median()
                })
    
    return pd.DataFrame(results)


def create_pivot_table(df, metric, agg_func='mean'):
    """Create pivot table: Rows=region, Cols=model, Values=metric."""
    pivot = df.pivot_table(
        values=metric,
        index='region',
        columns='model',
        aggfunc=agg_func
    )
    
    pivot['Mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean', ascending=False)
    
    return pivot


def create_visualizations(df, region_stats, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    metrics = get_evaluation_metrics()
    cultural_metric = 'Cultural Accuracy & Authenticity'
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Box plot: Cultural Accuracy by Region
    fig, ax = plt.subplots(figsize=(14, 8))
    
    order = df.groupby('region')[cultural_metric].mean().sort_values(ascending=False).index
    
    sns.boxplot(
        data=df,
        x='region',
        y=cultural_metric,
        order=order,
        palette='coolwarm',
        ax=ax
    )
    ax.set_title('RQ5: Cultural Accuracy by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Region', fontsize=11)
    ax.set_ylabel('Cultural Accuracy & Authenticity Score', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    overall_mean = df[cultural_metric].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_mean:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq5_boxplot_cultural_by_region.png', bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: Region × Model for Cultural Accuracy
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pivot = create_pivot_table(df, cultural_metric)
    pivot_for_heatmap = pivot.drop(columns=['Mean'], errors='ignore')
    
    sns.heatmap(
        pivot_for_heatmap,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=pivot_for_heatmap.values.mean(),
        ax=ax,
        cbar_kws={'label': 'Mean Score'}
    )
    ax.set_title('RQ5: Cultural Accuracy by Region × Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Region', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'rq5_heatmap_region_model.png', bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart: Binary comparison (Prominent vs Remote)
    df_cat = categorize_regions(df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    category_means = df_cat.groupby('region_category')[metrics].mean()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    metric_short = ['Cultural', 'Contextual', 'Narrative', 'Linguistic']
    
    bars1 = ax.bar(x - width/2, category_means.loc['Prominent'], width, label='Prominent', color='steelblue')
    bars2 = ax.bar(x + width/2, category_means.loc['Remote'], width, label='Remote', color='darkorange')
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ5: Prominent vs Remote Regions Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_short)
    ax.legend()
    ax.set_ylim(0, 5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq5_bar_prominent_vs_remote.png', bbox_inches='tight')
    plt.close()
    
    # 4. Regional ranking bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    region_overall = df.groupby('region')[metrics].mean().mean(axis=1).sort_values(ascending=True)
    
    colors = ['#e74c3c' if categorize_regions(df[df['region']==r]).iloc[0]['region_category'] == 'Remote' 
              else '#3498db' for r in region_overall.index]
    
    bars = ax.barh(range(len(region_overall)), region_overall.values, color=colors)
    ax.set_yticks(range(len(region_overall)))
    ax.set_yticklabels(region_overall.index)
    ax.set_xlabel('Overall Mean Score', fontsize=11)
    ax.set_ylabel('Region', fontsize=11)
    ax.set_title('RQ5: Regional Performance Ranking', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Prominent'),
                       Patch(facecolor='#e74c3c', label='Remote')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, region_overall.values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq5_regional_ranking.png', bbox_inches='tight')
    plt.close()
    
    # 5. All metrics by region grouped bar
    fig, ax = plt.subplots(figsize=(16, 8))
    
    region_order = df.groupby('region')[metrics].mean().mean(axis=1).sort_values(ascending=False).index
    region_means = df.groupby('region')[metrics].mean().loc[region_order]
    
    x = np.arange(len(region_order))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for i, (metric, short_name) in enumerate(zip(metrics, metric_short)):
        ax.bar(x + i*width, region_means[metric], width, label=short_name, color=colors[i])
    
    ax.set_xlabel('Region', fontsize=11)
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ5: All Metrics by Region', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(region_order, rotation=45, ha='right')
    ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq5_bar_all_metrics_by_region.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, region_stats, binary_comparison, anova_results):
    """Generate summary of key findings."""
    metrics = get_evaluation_metrics()
    cultural_metric = 'Cultural Accuracy & Authenticity'
    
    # Best and worst regions
    sorted_regions = region_stats.sort_values('overall_mean', ascending=False)
    best_region = sorted_regions.iloc[0]
    worst_region = sorted_regions.iloc[-1]
    
    # Binary comparison significance
    cultural_binary = binary_comparison[binary_comparison['metric'] == cultural_metric].iloc[0]
    
    summary = {
        'research_question': 'RQ5: Regional Contextual Performance',
        'main_question': 'Better performance on prominent vs remote regions?',
        'total_stories': len(df),
        'num_regions': df['region'].nunique(),
        'num_models': df['model'].nunique(),
        'regions': df['region'].unique().tolist(),
        
        # Best/worst regions
        'best_region': best_region['region'],
        'best_region_mean': best_region['overall_mean'],
        'worst_region': worst_region['region'],
        'worst_region_mean': worst_region['overall_mean'],
        'performance_gap': best_region['overall_mean'] - worst_region['overall_mean'],
        
        # Binary comparison
        'prominent_mean': cultural_binary['prominent_mean'],
        'remote_mean': cultural_binary['remote_mean'],
        'binary_t_statistic': cultural_binary['t_statistic'],
        'binary_p_value': cultural_binary['p_value'],
        'binary_cohens_d': cultural_binary['cohens_d'],
        'binary_significant': cultural_binary['significant'],
        'binary_favors': cultural_binary['favors'],
        
        # ANOVA
        'anova_f_cultural': anova_results[cultural_metric]['f_statistic'],
        'anova_p_cultural': anova_results[cultural_metric]['p_value'],
        'anova_eta_squared': anova_results[cultural_metric]['eta_squared'],
        'anova_significant': anova_results[cultural_metric]['significant']
    }
    
    return summary


def main():
    """Main execution function for RQ5 analysis."""
    print("=" * 70)
    print("RQ5: Regional Contextual Performance")
    print("=" * 70)
    print("\nQuestion: Is there better performance on prominent vs remote regions?")
    print("Method: Join CSV+Parquet for region, ANOVA + binary comparison")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load and merge data
    print("\n[1] Loading and merging data...")
    df_csv = load_csv_data()
    df_parquet = load_parquet_data()
    
    print(f"    CSV: {len(df_csv)} rows")
    print(f"    Parquet: {len(df_parquet)} rows")
    
    df = merge_data(df_csv, df_parquet)
    print(f"    Merged: {len(df)} rows with region")
    print(f"    Regions found: {df['region'].nunique()}")
    
    metrics = get_evaluation_metrics()
    cultural_metric = 'Cultural Accuracy & Authenticity'
    
    # Calculate region statistics
    print("\n[2] Calculating region statistics...")
    region_stats = calculate_region_stats(df, metrics)
    region_stats.to_csv(output_dir / 'rq5_region_stats.csv', index=False)
    print(f"    Exported: rq5_region_stats.csv")
    
    print("\n    Regional Ranking (by overall mean):")
    for _, row in region_stats.iterrows():
        print(f"    {int(row['rank']):2d}. {row['region']:<45} Mean: {row['overall_mean']:.3f} (n={int(row['n'])})")
    
    # Binary comparison: Prominent vs Remote
    print("\n[3] Performing binary comparison (Prominent vs Remote)...")
    binary_comparison = perform_binary_comparison(df, metrics)
    binary_comparison.to_csv(output_dir / 'rq5_binary_comparison.csv', index=False)
    print(f"    Exported: rq5_binary_comparison.csv")
    
    print("\n    Prominent vs Remote Comparison:")
    print("    " + "-" * 85)
    print(f"    {'Metric':<50} {'Prominent':>10} {'Remote':>10} {'p-value':>12} {'Sig?':>8}")
    print("    " + "-" * 85)
    for _, row in binary_comparison.iterrows():
        sig = '*' if row['significant'] else ''
        short_metric = row['metric'][:48] + '..' if len(row['metric']) > 50 else row['metric']
        print(f"    {short_metric:<50} {row['prominent_mean']:>10.3f} {row['remote_mean']:>10.3f} {row['p_value']:>12.6f} {sig:>8}")
    
    # One-way ANOVA for each metric
    print("\n[4] Performing ANOVA across all regions...")
    anova_results = {}
    anova_summary = []
    
    for metric in metrics:
        result = perform_anova_regions(df, metric)
        anova_results[metric] = result
        anova_summary.append({
            'metric': metric,
            'f_statistic': result['f_statistic'],
            'p_value': result['p_value'],
            'eta_squared': result['eta_squared'],
            'significant': result['significant']
        })
    
    anova_df = pd.DataFrame(anova_summary)
    anova_df.to_csv(output_dir / 'rq5_anova_regions.csv', index=False)
    print(f"    Exported: rq5_anova_regions.csv")
    
    print("\n    One-way ANOVA Results (Region effect):")
    print("    " + "-" * 85)
    print(f"    {'Metric':<50} {'F-stat':>10} {'p-value':>12} {'η²':>8} {'Sig?':>8}")
    print("    " + "-" * 85)
    for _, row in anova_df.iterrows():
        sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        short_metric = row['metric'][:48] + '..' if len(row['metric']) > 50 else row['metric']
        print(f"    {short_metric:<50} {row['f_statistic']:>10.3f} {row['p_value']:>12.6f} {row['eta_squared']:>8.4f} {sig_marker:>8}")
    
    # Tukey HSD post-hoc
    print("\n[5] Performing Tukey HSD post-hoc test...")
    tukey_results = perform_tukey_hsd(df, cultural_metric, 'region')
    
    if tukey_results is not None:
        tukey_results.to_csv(output_dir / 'rq5_tukey_hsd_regions.csv', index=False)
        print(f"    Exported: rq5_tukey_hsd_regions.csv")
        
        sig_pairs = tukey_results[tukey_results['reject'] == True] if 'reject' in tukey_results.columns else pd.DataFrame()
        if len(sig_pairs) > 0:
            print(f"\n    Significant pairwise differences: {len(sig_pairs)}")
            for _, row in sig_pairs.head(10).iterrows():
                print(f"      {row['group1'][:25]}.. vs {row['group2'][:25]}..: p={float(row['p-adj']):.4f}")
        else:
            print("    No significant pairwise differences at α=0.05")
    
    # Model × Region statistics
    print("\n[6] Calculating Model × Region statistics...")
    model_region_stats = calculate_model_by_region_stats(df, cultural_metric)
    model_region_stats.to_csv(output_dir / 'rq5_model_region_stats.csv', index=False)
    print(f"    Exported: rq5_model_region_stats.csv")
    
    # Pivot table
    print("\n[7] Creating pivot tables...")
    pivot_cultural = create_pivot_table(df, cultural_metric)
    pivot_cultural.to_csv(output_dir / 'rq5_pivot_cultural_region_model.csv')
    print(f"    Exported: rq5_pivot_cultural_region_model.csv")
    
    # Region category stats
    print("\n[8] Exporting region category mapping...")
    df_cat = categorize_regions(df)
    category_stats = df_cat.groupby('region_category')[metrics].agg(['mean', 'std', 'count'])
    category_stats.to_csv(output_dir / 'rq5_category_stats.csv')
    print(f"    Exported: rq5_category_stats.csv")
    
    # Create visualizations
    print("\n[9] Creating visualizations...")
    create_visualizations(df, region_stats, output_dir)
    
    # Generate summary
    print("\n[10] Generating summary...")
    summary = generate_summary(df, region_stats, binary_comparison, anova_results)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq5_summary.csv', index=False)
    print(f"    Exported: rq5_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ5 ANALYSIS COMPLETE")
    print("=" * 70)
    
    cultural_binary = binary_comparison[binary_comparison['metric'] == cultural_metric].iloc[0]
    cultural_anova = anova_results[cultural_metric]
    
    print(f"\nKey Findings:")
    print(f"\n  Binary Comparison (Prominent vs Remote):")
    print(f"    Prominent regions mean: {cultural_binary['prominent_mean']:.3f}")
    print(f"    Remote regions mean:    {cultural_binary['remote_mean']:.3f}")
    print(f"    Difference:             {cultural_binary['mean_difference']:.3f}")
    print(f"    t-statistic:            {cultural_binary['t_statistic']:.3f}")
    print(f"    p-value:                {cultural_binary['p_value']:.6f}")
    print(f"    Effect size (Cohen's d): {cultural_binary['cohens_d']:.4f}")
    
    if cultural_binary['significant']:
        print(f"\n  ⚠️  SIGNIFICANT REGIONAL BIAS DETECTED!")
        print(f"     {cultural_binary['favors']} regions show better performance.")
    else:
        print(f"\n  ✓  No significant regional bias detected (p ≥ 0.05)")
    
    print(f"\n  ANOVA across all 9 regions:")
    print(f"    F-statistic: {cultural_anova['f_statistic']:.3f}")
    print(f"    p-value:     {cultural_anova['p_value']:.6f}")
    print(f"    η²:          {cultural_anova['eta_squared']:.4f}")
    
    sorted_regions = region_stats.sort_values('overall_mean', ascending=False)
    print(f"\n  Best performing region:  {sorted_regions.iloc[0]['region']} ({sorted_regions.iloc[0]['overall_mean']:.3f})")
    print(f"  Worst performing region: {sorted_regions.iloc[-1]['region']} ({sorted_regions.iloc[-1]['overall_mean']:.3f})")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq5_region_stats.csv - Statistics for each region")
    print("  2. rq5_binary_comparison.csv - Prominent vs Remote comparison")
    print("  3. rq5_anova_regions.csv - ANOVA results")
    print("  4. rq5_tukey_hsd_regions.csv - Post-hoc comparisons")
    print("  5. rq5_model_region_stats.csv - Model × Region stats")
    print("  6. rq5_pivot_cultural_region_model.csv - Pivot table")
    print("  7. rq5_category_stats.csv - Prominent/Remote category stats")
    print("  8. rq5_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, region_stats, binary_comparison, anova_results


if __name__ == "__main__":
    df, region_stats, binary_comparison, anova_results = main()
