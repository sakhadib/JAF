"""
RQ3: Narrative Complexity vs. Model Performance
================================================
Question: Does story_type impact narrative coherence scores?
          Which story types are most challenging for LLMs to generate?

Method:
- Join CSV evaluations with parquet to get `story_type`
- Pivot table: Rows: story_type, Cols: models, Values: mean(Narrative Coherence)
- Two-way ANOVA (story_type × model interaction)
- Identify difficult story types (lowest scores)
- Visualization: Heatmap of story_type × model × coherence

Required Join: CSV + Parquet on (model, culture, story)
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

# Try to import statsmodels for advanced ANOVA
try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Using basic ANOVA only.")


def load_csv_data():
    """Load the evaluation dataset from CSV."""
    data_path = Path(__file__).parent.parent / "data" / "evaluated_stories.csv"
    df = pd.read_csv(data_path)
    return df


def load_parquet_data():
    """Load parquet data to get story_type column."""
    parquet_dir = Path(__file__).parent.parent / "data" / "parquet"
    
    # Use one parquet file - they all have the same story metadata
    # Using openai-large as reference
    parquet_file = parquet_dir / "openai_text-embedding-3-large.parquet"
    
    if not parquet_file.exists():
        # Try any available parquet file
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/parquet/")
        parquet_file = parquet_files[0]
    
    df = pd.read_parquet(parquet_file)
    
    # Keep only columns needed for joining
    return df[['model', 'culture', 'story', 'story_type', 'region']]


def merge_data(df_csv, df_parquet):
    """
    Merge CSV evaluations with parquet metadata to get story_type.
    Join key: (model, culture, story)
    """
    # Perform the merge
    df_merged = pd.merge(
        df_csv,
        df_parquet[['model', 'culture', 'story', 'story_type']],
        on=['model', 'culture', 'story'],
        how='left'
    )
    
    # Validate merge
    missing = df_merged['story_type'].isna().sum()
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


def calculate_story_type_stats(df, metrics):
    """Calculate descriptive statistics for each story_type across all metrics."""
    results = []
    
    for story_type in df['story_type'].unique():
        type_data = df[df['story_type'] == story_type]
        row = {
            'story_type': story_type,
            'n': len(type_data),
            'n_models': type_data['model'].nunique(),
            'n_cultures': type_data['culture'].nunique()
        }
        
        for metric in metrics:
            row[f'{metric}_mean'] = type_data[metric].mean()
            row[f'{metric}_std'] = type_data[metric].std()
            row[f'{metric}_median'] = type_data[metric].median()
        
        # Calculate overall average across all metrics
        row['overall_mean'] = np.mean([row[f'{m}_mean'] for m in metrics])
        results.append(row)
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values('overall_mean', ascending=False)
    stats_df['difficulty_rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def create_pivot_table(df, metric, agg_func='mean'):
    """
    Create pivot table: Rows=story_type, Cols=model, Values=metric.
    """
    pivot = df.pivot_table(
        values=metric,
        index='story_type',
        columns='model',
        aggfunc=agg_func
    )
    
    # Add row means and sort by them
    pivot['Mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean', ascending=False)
    
    return pivot


def perform_one_way_anova_story_type(df, metric):
    """Perform one-way ANOVA testing story_type effect on metric."""
    story_types = df['story_type'].unique()
    groups = [df[df['story_type'] == st][metric].dropna() for st in story_types]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'f_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    # SS_between / SS_total
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


def perform_two_way_anova(df, metric):
    """
    Perform two-way ANOVA: story_type × model interaction.
    Tests main effects and interaction.
    """
    if not HAS_STATSMODELS:
        return None
    
    # Clean column name for formula
    metric_clean = metric.replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
    df_temp = df.copy()
    df_temp[metric_clean] = df_temp[metric]
    
    # Fit the two-way ANOVA model
    try:
        formula = f'{metric_clean} ~ C(story_type) + C(model) + C(story_type):C(model)'
        model = ols(formula, data=df_temp).fit()
        anova_table = anova_lm(model, typ=2)
        
        return anova_table
    except Exception as e:
        print(f"Two-way ANOVA failed for {metric}: {e}")
        return None


def perform_tukey_hsd(df, metric, group_col='story_type'):
    """Perform Tukey HSD post-hoc test for pairwise comparisons."""
    if not HAS_STATSMODELS:
        return None
    
    try:
        tukey = pairwise_tukeyhsd(
            endog=df[metric].dropna(),
            groups=df.loc[df[metric].notna(), group_col],
            alpha=0.05
        )
        
        # Convert to DataFrame
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


def calculate_model_by_story_type_stats(df, metric):
    """Calculate statistics for each model × story_type combination."""
    results = []
    
    for model in df['model'].unique():
        for story_type in df['story_type'].unique():
            subset = df[(df['model'] == model) & (df['story_type'] == story_type)]
            if len(subset) > 0:
                results.append({
                    'model': model,
                    'story_type': story_type,
                    'n': len(subset),
                    'mean': subset[metric].mean(),
                    'std': subset[metric].std(),
                    'median': subset[metric].median()
                })
    
    return pd.DataFrame(results)


def identify_difficult_story_types(df, metric, top_n=3):
    """Identify the most challenging story types (lowest scores)."""
    type_means = df.groupby('story_type')[metric].agg(['mean', 'std', 'count'])
    type_means = type_means.sort_values('mean')
    
    difficult = type_means.head(top_n).reset_index()
    difficult.columns = ['story_type', 'mean', 'std', 'count']
    difficult['rank'] = range(1, len(difficult) + 1)
    
    return difficult


def identify_easy_story_types(df, metric, top_n=3):
    """Identify the easiest story types (highest scores)."""
    type_means = df.groupby('story_type')[metric].agg(['mean', 'std', 'count'])
    type_means = type_means.sort_values('mean', ascending=False)
    
    easy = type_means.head(top_n).reset_index()
    easy.columns = ['story_type', 'mean', 'std', 'count']
    easy['rank'] = range(1, len(easy) + 1)
    
    return easy


def create_visualizations(df, pivot_coherence, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    metrics = get_evaluation_metrics()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Heatmap: Story Type × Model for Narrative Coherence
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Remove 'Mean' column for heatmap
    pivot_for_heatmap = pivot_coherence.drop(columns=['Mean'], errors='ignore')
    
    sns.heatmap(
        pivot_for_heatmap,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=pivot_for_heatmap.values.mean(),
        ax=ax,
        cbar_kws={'label': 'Mean Score'}
    )
    ax.set_title('RQ3: Narrative Coherence by Story Type × Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Story Type', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_heatmap_coherence_storytype_model.png', bbox_inches='tight')
    plt.close()
    
    # 2. Box plot: Story Type effect on Narrative Coherence
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Order story types by mean coherence
    coherence_metric = 'Narrative & Symbolic Coherence'
    order = df.groupby('story_type')[coherence_metric].mean().sort_values(ascending=False).index
    
    sns.boxplot(
        data=df,
        x='story_type',
        y=coherence_metric,
        order=order,
        palette='coolwarm',
        ax=ax
    )
    ax.set_title('RQ3: Narrative Coherence Distribution by Story Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Story Type', fontsize=11)
    ax.set_ylabel('Narrative & Symbolic Coherence Score', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean line
    overall_mean = df[coherence_metric].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_mean:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_boxplot_coherence_by_storytype.png', bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart: Average scores across all metrics by story type
    fig, ax = plt.subplots(figsize=(14, 8))
    
    type_means = df.groupby('story_type')[metrics].mean()
    type_means = type_means.loc[order]  # Same order as above
    
    x = np.arange(len(type_means))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    metric_short = ['Cultural', 'Contextual', 'Narrative', 'Linguistic']
    
    for i, (metric, short_name) in enumerate(zip(metrics, metric_short)):
        ax.bar(x + i*width, type_means[metric], width, label=short_name, color=colors[i])
    
    ax.set_xlabel('Story Type', fontsize=11)
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_title('RQ3: All Evaluation Metrics by Story Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(type_means.index, rotation=45, ha='right')
    ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_bar_all_metrics_by_storytype.png', bbox_inches='tight')
    plt.close()
    
    # 4. Interaction plot: Model × Story Type
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate means for each model-story_type combination
    interaction_df = df.groupby(['model', 'story_type'])[coherence_metric].mean().reset_index()
    
    # Plot lines for each model
    models = df['model'].unique()
    palette = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        model_data = interaction_df[interaction_df['model'] == model]
        model_data = model_data.set_index('story_type').reindex(order)
        ax.plot(range(len(order)), model_data[coherence_metric].values, 
                marker='o', label=model, color=palette[i], linewidth=2, markersize=8)
    
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_xlabel('Story Type', fontsize=11)
    ax.set_ylabel('Mean Narrative Coherence', fontsize=11)
    ax.set_title('RQ3: Model × Story Type Interaction', fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_interaction_model_storytype.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, story_type_stats, anova_results, two_way_anova):
    """Generate a summary of key findings."""
    metrics = get_evaluation_metrics()
    coherence_metric = 'Narrative & Symbolic Coherence'
    
    # Find easiest and hardest story types
    sorted_types = story_type_stats.sort_values('overall_mean', ascending=False)
    easiest = sorted_types.iloc[0]
    hardest = sorted_types.iloc[-1]
    
    # Check significance
    anova_coherence = anova_results[coherence_metric]
    
    summary = {
        'research_question': 'RQ3: Narrative Complexity vs. Model Performance',
        'main_question': 'Does story_type impact narrative coherence scores?',
        'total_stories': len(df),
        'num_story_types': df['story_type'].nunique(),
        'num_models': df['model'].nunique(),
        'story_types': df['story_type'].unique().tolist(),
        
        # Key findings for coherence
        'coherence_anova_f': anova_coherence['f_statistic'],
        'coherence_anova_p': anova_coherence['p_value'],
        'coherence_effect_size': anova_coherence['eta_squared'],
        'coherence_significant': anova_coherence['significant'],
        
        # Easiest/hardest story types
        'easiest_story_type': easiest['story_type'],
        'easiest_mean': easiest['overall_mean'],
        'hardest_story_type': hardest['story_type'],
        'hardest_mean': hardest['overall_mean'],
        'performance_gap': easiest['overall_mean'] - hardest['overall_mean'],
        
        # Two-way ANOVA results
        'interaction_tested': two_way_anova is not None
    }
    
    if two_way_anova is not None:
        try:
            # Extract interaction p-value
            interaction_row = [idx for idx in two_way_anova.index if 'story_type' in str(idx) and 'model' in str(idx)]
            if interaction_row:
                summary['interaction_p_value'] = float(two_way_anova.loc[interaction_row[0], 'PR(>F)'])
                summary['interaction_significant'] = summary['interaction_p_value'] < 0.05
        except:
            pass
    
    return summary


def main():
    """Main execution function for RQ3 analysis."""
    print("=" * 70)
    print("RQ3: Narrative Complexity vs. Model Performance")
    print("=" * 70)
    print("\nQuestion: Does story_type impact narrative coherence scores?")
    print("Method: Join CSV+Parquet, Two-way ANOVA (story_type × model)")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading and merging data...")
    df_csv = load_csv_data()
    df_parquet = load_parquet_data()
    
    print(f"    CSV: {len(df_csv)} rows")
    print(f"    Parquet: {len(df_parquet)} rows")
    
    # Merge to get story_type
    df = merge_data(df_csv, df_parquet)
    print(f"    Merged: {len(df)} rows with story_type")
    print(f"    Story types found: {df['story_type'].nunique()}")
    
    metrics = get_evaluation_metrics()
    coherence_metric = 'Narrative & Symbolic Coherence'
    
    # Calculate story type statistics
    print("\n[2] Calculating story type statistics...")
    story_type_stats = calculate_story_type_stats(df, metrics)
    story_type_stats.to_csv(output_dir / 'rq3_story_type_stats.csv', index=False)
    print(f"    Exported: rq3_story_type_stats.csv")
    
    print("\n    Story Type Ranking (by overall mean):")
    for _, row in story_type_stats.iterrows():
        print(f"    {int(row['difficulty_rank']):2d}. {row['story_type']:<40} Mean: {row['overall_mean']:.3f} (n={int(row['n'])})")
    
    # Create pivot tables for each metric
    print("\n[3] Creating pivot tables (Story Type × Model)...")
    pivot_tables = {}
    for metric in metrics:
        pivot = create_pivot_table(df, metric)
        pivot_tables[metric] = pivot
    
    # Save coherence pivot table
    pivot_coherence = pivot_tables[coherence_metric]
    pivot_coherence.to_csv(output_dir / 'rq3_pivot_coherence_storytype_model.csv')
    print(f"    Exported: rq3_pivot_coherence_storytype_model.csv")
    
    # Save all metrics pivot
    all_pivots_list = []
    for metric, pivot in pivot_tables.items():
        pivot_flat = pivot.reset_index()
        pivot_flat['metric'] = metric
        all_pivots_list.append(pivot_flat)
    all_pivots_df = pd.concat(all_pivots_list, ignore_index=True)
    all_pivots_df.to_csv(output_dir / 'rq3_pivot_all_metrics.csv', index=False)
    print(f"    Exported: rq3_pivot_all_metrics.csv")
    
    # One-way ANOVA for story_type effect
    print("\n[4] Performing One-way ANOVA (story_type effect)...")
    anova_results = {}
    anova_summary = []
    
    for metric in metrics:
        result = perform_one_way_anova_story_type(df, metric)
        anova_results[metric] = result
        anova_summary.append({
            'metric': metric,
            'f_statistic': result['f_statistic'],
            'p_value': result['p_value'],
            'eta_squared': result['eta_squared'],
            'significant': result['significant']
        })
    
    anova_df = pd.DataFrame(anova_summary)
    anova_df.to_csv(output_dir / 'rq3_anova_storytype.csv', index=False)
    print(f"    Exported: rq3_anova_storytype.csv")
    
    print("\n    One-way ANOVA Results (Story Type effect):")
    print("    " + "-" * 85)
    print(f"    {'Metric':<50} {'F-stat':>10} {'p-value':>12} {'η²':>8} {'Sig?':>8}")
    print("    " + "-" * 85)
    for _, row in anova_df.iterrows():
        sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        print(f"    {row['metric']:<50} {row['f_statistic']:>10.3f} {row['p_value']:>12.6f} {row['eta_squared']:>8.4f} {sig_marker:>8}")
    
    # Two-way ANOVA (story_type × model interaction)
    print("\n[5] Performing Two-way ANOVA (story_type × model)...")
    two_way_anova = perform_two_way_anova(df, coherence_metric)
    
    if two_way_anova is not None:
        two_way_anova.to_csv(output_dir / 'rq3_twoway_anova_coherence.csv')
        print(f"    Exported: rq3_twoway_anova_coherence.csv")
        print("\n    Two-way ANOVA Results for Narrative Coherence:")
        print(two_way_anova.to_string())
    else:
        print("    Two-way ANOVA not available (statsmodels required)")
    
    # Tukey HSD post-hoc for story_type
    print("\n[6] Performing Tukey HSD post-hoc test...")
    tukey_results = perform_tukey_hsd(df, coherence_metric, 'story_type')
    
    if tukey_results is not None:
        tukey_results.to_csv(output_dir / 'rq3_tukey_hsd_storytype.csv', index=False)
        print(f"    Exported: rq3_tukey_hsd_storytype.csv")
        
        # Show significant pairs
        sig_pairs = tukey_results[tukey_results['reject'] == True] if 'reject' in tukey_results.columns else pd.DataFrame()
        if len(sig_pairs) > 0:
            print(f"\n    Significant pairwise differences found: {len(sig_pairs)}")
            for _, row in sig_pairs.head(10).iterrows():
                print(f"      {row['group1']} vs {row['group2']}: p={float(row['p-adj']):.4f}")
        else:
            print("    No significant pairwise differences at α=0.05")
    
    # Model × Story Type detailed stats
    print("\n[7] Calculating Model × Story Type statistics...")
    model_storytype_stats = calculate_model_by_story_type_stats(df, coherence_metric)
    model_storytype_stats.to_csv(output_dir / 'rq3_model_storytype_stats.csv', index=False)
    print(f"    Exported: rq3_model_storytype_stats.csv")
    
    # Identify difficult/easy story types
    print("\n[8] Identifying challenging and easy story types...")
    
    difficult = identify_difficult_story_types(df, coherence_metric, top_n=3)
    easy = identify_easy_story_types(df, coherence_metric, top_n=3)
    
    print("\n    MOST CHALLENGING Story Types (lowest coherence):")
    for _, row in difficult.iterrows():
        print(f"      {int(row['rank'])}. {row['story_type']:<35} Mean: {row['mean']:.3f} ± {row['std']:.3f}")
    
    print("\n    EASIEST Story Types (highest coherence):")
    for _, row in easy.iterrows():
        print(f"      {int(row['rank'])}. {row['story_type']:<35} Mean: {row['mean']:.3f} ± {row['std']:.3f}")
    
    # Combine into difficulty ranking export
    difficulty_ranking = df.groupby('story_type')[coherence_metric].agg(['mean', 'std', 'count']).reset_index()
    difficulty_ranking = difficulty_ranking.sort_values('mean')
    difficulty_ranking['difficulty_rank'] = range(1, len(difficulty_ranking) + 1)
    difficulty_ranking.columns = ['story_type', 'coherence_mean', 'coherence_std', 'n', 'difficulty_rank']
    difficulty_ranking.to_csv(output_dir / 'rq3_difficulty_ranking.csv', index=False)
    print(f"\n    Exported: rq3_difficulty_ranking.csv")
    
    # Create visualizations
    print("\n[9] Creating visualizations...")
    create_visualizations(df, pivot_coherence, output_dir)
    
    # Generate and save summary
    print("\n[10] Generating summary...")
    summary = generate_summary(df, story_type_stats, anova_results, two_way_anova)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq3_summary.csv', index=False)
    print(f"    Exported: rq3_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ3 ANALYSIS COMPLETE")
    print("=" * 70)
    
    coherence_result = anova_results[coherence_metric]
    print(f"\nKey Finding: Story type effect on Narrative Coherence")
    print(f"  - F-statistic: {coherence_result['f_statistic']:.3f}")
    print(f"  - p-value: {coherence_result['p_value']:.6f}")
    print(f"  - Effect size (η²): {coherence_result['eta_squared']:.4f}")
    
    if coherence_result['significant']:
        print(f"  - Result: SIGNIFICANT (p < 0.05)")
        print(f"\n  Story type significantly impacts narrative coherence!")
    else:
        print(f"  - Result: NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"\n  No significant effect of story type on coherence.")
    
    print(f"\n  Most challenging type: {difficult.iloc[0]['story_type']} ({difficult.iloc[0]['mean']:.3f})")
    print(f"  Easiest type: {easy.iloc[0]['story_type']} ({easy.iloc[0]['mean']:.3f})")
    print(f"  Performance gap: {easy.iloc[0]['mean'] - difficult.iloc[0]['mean']:.3f}")
    
    if two_way_anova is not None:
        try:
            interaction_row = [idx for idx in two_way_anova.index if 'story_type' in str(idx) and 'model' in str(idx)]
            if interaction_row:
                interaction_p = float(two_way_anova.loc[interaction_row[0], 'PR(>F)'])
                print(f"\n  Model × Story Type Interaction:")
                print(f"  - p-value: {interaction_p:.6f}")
                if interaction_p < 0.05:
                    print(f"  - SIGNIFICANT interaction: Some models handle certain types better")
                else:
                    print(f"  - No significant interaction: Models perform consistently across types")
        except:
            pass
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq3_story_type_stats.csv - Statistics for each story type")
    print("  2. rq3_pivot_coherence_storytype_model.csv - Pivot table")
    print("  3. rq3_pivot_all_metrics.csv - All metrics pivot")
    print("  4. rq3_anova_storytype.csv - One-way ANOVA results")
    print("  5. rq3_twoway_anova_coherence.csv - Two-way ANOVA results")
    print("  6. rq3_tukey_hsd_storytype.csv - Post-hoc comparisons")
    print("  7. rq3_model_storytype_stats.csv - Model × Story Type stats")
    print("  8. rq3_difficulty_ranking.csv - Difficulty ranking")
    print("  9. rq3_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, story_type_stats, anova_results, two_way_anova


if __name__ == "__main__":
    df, story_type_stats, anova_results, two_way_anova = main()
