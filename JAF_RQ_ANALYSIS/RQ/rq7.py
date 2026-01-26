"""
RQ7: Prompt Adherence Quantification
=====================================
Question: Which generation model stays truest to the prompt?
          How well do generated stories align with their scenario prompts?

Method:
- Compute cosine_similarity(scenario_embedding, story_embedding) per row
- Group by `model` to get average similarity per generation model
- Statistical test: ANOVA on similarity scores
- Visualize: Box plot of adherence by model

Data Source: Parquet only (scenario_embedding + story_embedding)
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

# Try to import statsmodels for Tukey HSD
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Skipping Tukey HSD test.")


def load_parquet_data(embedding_model='openai_text-embedding-3-large'):
    """Load parquet data with scenario and story embeddings."""
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


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    # cosine distance = 1 - similarity, so similarity = 1 - distance
    return 1 - cosine(vec1, vec2)


def compute_prompt_adherence(df):
    """
    Compute prompt adherence (cosine similarity) between 
    scenario_embedding and story_embedding for each row.
    """
    similarities = []
    
    for idx, row in df.iterrows():
        scenario_emb = np.array(row['scenario_embedding'])
        story_emb = np.array(row['story_embedding'])
        
        sim = cosine_similarity(scenario_emb, story_emb)
        similarities.append(sim)
    
    df = df.copy()
    df['prompt_adherence'] = similarities
    
    return df


def calculate_model_adherence_stats(df):
    """Calculate prompt adherence statistics grouped by model."""
    stats_df = df.groupby('model')['prompt_adherence'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    # Add IQR
    stats_df['iqr'] = stats_df['q75'] - stats_df['q25']
    
    # Sort by mean adherence (descending = best first)
    stats_df = stats_df.sort_values('mean', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def calculate_culture_adherence_stats(df):
    """Calculate prompt adherence statistics grouped by culture."""
    stats_df = df.groupby('culture')['prompt_adherence'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).reset_index()
    
    stats_df = stats_df.sort_values('mean', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def calculate_story_type_adherence_stats(df):
    """Calculate prompt adherence statistics grouped by story_type."""
    stats_df = df.groupby('story_type')['prompt_adherence'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).reset_index()
    
    stats_df = stats_df.sort_values('mean', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    return stats_df


def perform_anova_models(df):
    """Perform one-way ANOVA testing model effect on prompt adherence."""
    models = df['model'].unique()
    groups = [df[df['model'] == m]['prompt_adherence'].dropna() for m in models]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'f_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    grand_mean = df['prompt_adherence'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df['prompt_adherence'] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < 0.05
    }


def perform_tukey_hsd(df):
    """Perform Tukey HSD post-hoc test for pairwise model comparisons."""
    if not HAS_STATSMODELS:
        return None
    
    try:
        tukey = pairwise_tukeyhsd(
            endog=df['prompt_adherence'].dropna(),
            groups=df.loc[df['prompt_adherence'].notna(), 'model'],
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


def calculate_model_culture_stats(df):
    """Calculate adherence for each model × culture combination."""
    stats_df = df.groupby(['model', 'culture'])['prompt_adherence'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std')
    ]).reset_index()
    
    return stats_df


def create_pivot_table(df):
    """Create pivot table: Rows=model, Cols=culture, Values=mean adherence."""
    pivot = df.pivot_table(
        values='prompt_adherence',
        index='model',
        columns='culture',
        aggfunc='mean'
    )
    
    pivot['Overall'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Overall', ascending=False)
    
    return pivot


def identify_best_worst_combinations(df):
    """Identify best and worst model-culture-story_type combinations."""
    combo_stats = df.groupby(['model', 'culture', 'story_type'])['prompt_adherence'].agg([
        ('n', 'count'),
        ('mean', 'mean')
    ]).reset_index()
    
    combo_stats = combo_stats.sort_values('mean', ascending=False)
    
    best = combo_stats.head(10)
    worst = combo_stats.tail(10)
    
    return best, worst


def create_visualizations(df, model_stats, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Box plot: Prompt adherence by model
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Order by mean adherence
    order = model_stats.sort_values('mean', ascending=False)['model'].tolist()
    
    sns.boxplot(
        data=df,
        x='model',
        y='prompt_adherence',
        order=order,
        palette='viridis',
        ax=ax
    )
    ax.set_title('RQ7: Prompt Adherence by Generation Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Prompt Adherence (Cosine Similarity)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    # Add overall mean line
    overall_mean = df['prompt_adherence'].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {overall_mean:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq7_boxplot_adherence_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 2. Violin plot for more detail
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.violinplot(
        data=df,
        x='model',
        y='prompt_adherence',
        order=order,
        palette='viridis',
        ax=ax
    )
    ax.set_title('RQ7: Prompt Adherence Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Prompt Adherence (Cosine Similarity)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq7_violin_adherence_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_order = model_stats.sort_values('mean', ascending=False)
    
    bars = ax.bar(
        range(len(model_order)),
        model_order['mean'],
        yerr=model_order['std'],
        capsize=5,
        color='steelblue',
        edgecolor='black'
    )
    
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order['model'], rotation=45, ha='right')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Mean Prompt Adherence', fontsize=11)
    ax.set_title('RQ7: Model Ranking by Prompt Adherence', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, model_order['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + model_order['std'].iloc[i] + 0.002,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq7_bar_adherence_ranking.png', bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap: Model × Culture adherence
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot = create_pivot_table(df)
    pivot_for_heatmap = pivot.drop(columns=['Overall'], errors='ignore')
    
    sns.heatmap(
        pivot_for_heatmap,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=pivot_for_heatmap.values.mean(),
        ax=ax,
        cbar_kws={'label': 'Prompt Adherence'}
    )
    ax.set_title('RQ7: Prompt Adherence by Model × Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq7_heatmap_model_culture.png', bbox_inches='tight')
    plt.close()
    
    # 5. Distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]['prompt_adherence']
        ax.hist(model_data, bins=30, alpha=0.5, label=model)
    
    ax.set_xlabel('Prompt Adherence', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('RQ7: Prompt Adherence Distribution by Model', fontsize=14, fontweight='bold')
    ax.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq7_histogram_adherence.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, model_stats, anova_result):
    """Generate summary of key findings."""
    best_model = model_stats.iloc[0]
    worst_model = model_stats.iloc[-1]
    
    summary = {
        'research_question': 'RQ7: Prompt Adherence Quantification',
        'main_question': 'Which generation model stays truest to the prompt?',
        'total_stories': len(df),
        'num_models': df['model'].nunique(),
        'num_cultures': df['culture'].nunique(),
        
        # Overall statistics
        'overall_mean_adherence': df['prompt_adherence'].mean(),
        'overall_std_adherence': df['prompt_adherence'].std(),
        'overall_min_adherence': df['prompt_adherence'].min(),
        'overall_max_adherence': df['prompt_adherence'].max(),
        
        # Best and worst models
        'best_model': best_model['model'],
        'best_model_mean': best_model['mean'],
        'best_model_std': best_model['std'],
        'worst_model': worst_model['model'],
        'worst_model_mean': worst_model['mean'],
        'worst_model_std': worst_model['std'],
        'adherence_gap': best_model['mean'] - worst_model['mean'],
        
        # ANOVA results
        'anova_f_statistic': anova_result['f_statistic'],
        'anova_p_value': anova_result['p_value'],
        'anova_eta_squared': anova_result['eta_squared'],
        'anova_significant': anova_result['significant']
    }
    
    return summary


def main():
    """Main execution function for RQ7 analysis."""
    print("=" * 70)
    print("RQ7: Prompt Adherence Quantification")
    print("=" * 70)
    print("\nQuestion: Which generation model stays truest to the prompt?")
    print("Method: cosine_similarity(scenario_embedding, story_embedding)")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading embedding data...")
    df = load_parquet_data('openai_text-embedding-3-large')
    print(f"    Loaded {len(df)} stories")
    print(f"    Models: {df['model'].nunique()}")
    print(f"    Cultures: {df['culture'].nunique()}")
    
    # Compute prompt adherence
    print("\n[2] Computing prompt adherence scores...")
    df = compute_prompt_adherence(df)
    print(f"    Overall mean adherence: {df['prompt_adherence'].mean():.4f}")
    print(f"    Overall std adherence:  {df['prompt_adherence'].std():.4f}")
    print(f"    Range: [{df['prompt_adherence'].min():.4f}, {df['prompt_adherence'].max():.4f}]")
    
    # Export raw adherence scores
    adherence_df = df[['model', 'culture', 'story_type', 'prompt_adherence']].copy()
    adherence_df.to_csv(output_dir / 'rq7_adherence_scores.csv', index=False)
    print(f"    Exported: rq7_adherence_scores.csv")
    
    # Calculate model statistics
    print("\n[3] Calculating model adherence statistics...")
    model_stats = calculate_model_adherence_stats(df)
    model_stats.to_csv(output_dir / 'rq7_model_stats.csv', index=False)
    print(f"    Exported: rq7_model_stats.csv")
    
    print("\n    Model Ranking by Prompt Adherence:")
    print("    " + "-" * 70)
    print(f"    {'Rank':<6} {'Model':<40} {'Mean':>10} {'Std':>10}")
    print("    " + "-" * 70)
    for _, row in model_stats.iterrows():
        print(f"    {int(row['rank']):<6} {row['model']:<40} {row['mean']:>10.4f} {row['std']:>10.4f}")
    
    # Calculate culture statistics
    print("\n[4] Calculating culture adherence statistics...")
    culture_stats = calculate_culture_adherence_stats(df)
    culture_stats.to_csv(output_dir / 'rq7_culture_stats.csv', index=False)
    print(f"    Exported: rq7_culture_stats.csv")
    
    # Calculate story type statistics
    print("\n[5] Calculating story type adherence statistics...")
    story_type_stats = calculate_story_type_adherence_stats(df)
    story_type_stats.to_csv(output_dir / 'rq7_story_type_stats.csv', index=False)
    print(f"    Exported: rq7_story_type_stats.csv")
    
    # Perform ANOVA
    print("\n[6] Performing ANOVA test...")
    anova_result = perform_anova_models(df)
    
    anova_df = pd.DataFrame([anova_result])
    anova_df.to_csv(output_dir / 'rq7_anova_models.csv', index=False)
    print(f"    Exported: rq7_anova_models.csv")
    
    print(f"\n    ANOVA Results:")
    print(f"    F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"    p-value:     {anova_result['p_value']:.6f}")
    print(f"    η² (effect): {anova_result['eta_squared']:.4f}")
    sig_marker = '***' if anova_result['p_value'] < 0.001 else '**' if anova_result['p_value'] < 0.01 else '*' if anova_result['p_value'] < 0.05 else ''
    print(f"    Significant: {'Yes' if anova_result['significant'] else 'No'} {sig_marker}")
    
    # Tukey HSD post-hoc
    print("\n[7] Performing Tukey HSD post-hoc test...")
    tukey_results = perform_tukey_hsd(df)
    
    if tukey_results is not None:
        tukey_results.to_csv(output_dir / 'rq7_tukey_hsd.csv', index=False)
        print(f"    Exported: rq7_tukey_hsd.csv")
        
        sig_pairs = tukey_results[tukey_results['reject'] == True] if 'reject' in tukey_results.columns else pd.DataFrame()
        if len(sig_pairs) > 0:
            print(f"\n    Significant pairwise differences: {len(sig_pairs)}")
            for _, row in sig_pairs.head(10).iterrows():
                print(f"      {row['group1']} vs {row['group2']}: p={float(row['p-adj']):.4f}")
        else:
            print("    No significant pairwise differences at α=0.05")
    
    # Model × Culture statistics
    print("\n[8] Calculating Model × Culture statistics...")
    model_culture_stats = calculate_model_culture_stats(df)
    model_culture_stats.to_csv(output_dir / 'rq7_model_culture_stats.csv', index=False)
    print(f"    Exported: rq7_model_culture_stats.csv")
    
    # Pivot table
    pivot = create_pivot_table(df)
    pivot.to_csv(output_dir / 'rq7_pivot_model_culture.csv')
    print(f"    Exported: rq7_pivot_model_culture.csv")
    
    # Best/worst combinations
    print("\n[9] Identifying best/worst combinations...")
    best_combos, worst_combos = identify_best_worst_combinations(df)
    best_combos.to_csv(output_dir / 'rq7_best_combinations.csv', index=False)
    worst_combos.to_csv(output_dir / 'rq7_worst_combinations.csv', index=False)
    print(f"    Exported: rq7_best_combinations.csv")
    print(f"    Exported: rq7_worst_combinations.csv")
    
    # Create visualizations
    print("\n[10] Creating visualizations...")
    create_visualizations(df, model_stats, output_dir)
    
    # Generate summary
    print("\n[11] Generating summary...")
    summary = generate_summary(df, model_stats, anova_result)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq7_summary.csv', index=False)
    print(f"    Exported: rq7_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ7 ANALYSIS COMPLETE")
    print("=" * 70)
    
    best = model_stats.iloc[0]
    worst = model_stats.iloc[-1]
    
    print(f"\nKey Findings:")
    print(f"\n  Overall Prompt Adherence: {df['prompt_adherence'].mean():.4f} ± {df['prompt_adherence'].std():.4f}")
    
    print(f"\n  Model Ranking:")
    print(f"    🥇 Best:  {best['model']} ({best['mean']:.4f})")
    print(f"    🥉 Worst: {worst['model']} ({worst['mean']:.4f})")
    print(f"    Gap:     {best['mean'] - worst['mean']:.4f}")
    
    if anova_result['significant']:
        print(f"\n  ✓ SIGNIFICANT difference between models (p < 0.05)")
        print(f"    Models vary significantly in how well they follow prompts.")
    else:
        print(f"\n  ✗ No significant difference between models (p ≥ 0.05)")
        print(f"    All models follow prompts with similar fidelity.")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq7_adherence_scores.csv - Raw adherence scores per story")
    print("  2. rq7_model_stats.csv - Model statistics")
    print("  3. rq7_culture_stats.csv - Culture statistics")
    print("  4. rq7_story_type_stats.csv - Story type statistics")
    print("  5. rq7_anova_models.csv - ANOVA results")
    print("  6. rq7_tukey_hsd.csv - Post-hoc comparisons")
    print("  7. rq7_model_culture_stats.csv - Model × Culture stats")
    print("  8. rq7_pivot_model_culture.csv - Pivot table")
    print("  9. rq7_best_combinations.csv - Best performing combinations")
    print(" 10. rq7_worst_combinations.csv - Worst performing combinations")
    print(" 11. rq7_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, model_stats, anova_result


if __name__ == "__main__":
    df, model_stats, anova_result = main()
