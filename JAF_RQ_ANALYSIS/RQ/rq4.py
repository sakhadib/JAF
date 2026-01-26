"""
RQ4: Linguistic Fluency vs. Cultural Accuracy Correlation
==========================================================
Question: Are models "hallucinating fluently"?
          Is there a correlation between linguistic fluency and cultural accuracy?

Method:
- Extract Linguistic_Appropriateness and Cultural_Accuracy columns
- Pearson + Spearman correlation coefficients
- Scatter plot with regression line and confidence interval
- Per-model correlation analysis

Interpretation Guide:
- r > 0.7: Strong link (fluency implies accuracy)
- r < 0.3: Weak link (models hallucinate fluently)

Data Source: CSV only (no join required)
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

# Try to import statsmodels for regression
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Limited regression analysis.")


def load_data():
    """Load the evaluation dataset."""
    data_path = Path(__file__).parent.parent / "data" / "evaluated_stories.csv"
    df = pd.read_csv(data_path)
    return df


def get_key_metrics():
    """Return the two key metrics for this analysis."""
    return {
        'linguistic': 'Linguistic & Expressive Appropriateness (Bangla)',
        'cultural': 'Cultural Accuracy & Authenticity'
    }


def get_all_metrics():
    """Return all evaluation metric columns."""
    return [
        'Cultural Accuracy & Authenticity',
        'Contextual & Temporal Appropriateness',
        'Narrative & Symbolic Coherence',
        'Linguistic & Expressive Appropriateness (Bangla)'
    ]


def calculate_overall_correlation(df, metric1, metric2):
    """Calculate Pearson and Spearman correlations between two metrics."""
    x = df[metric1].dropna()
    y = df[metric2].dropna()
    
    # Align indices
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Coefficient of determination
    r_squared = pearson_r ** 2
    
    return {
        'n': len(x),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r_squared
    }


def calculate_per_model_correlation(df, metric1, metric2):
    """Calculate correlations for each model separately."""
    results = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        x = model_data[metric1].dropna()
        y = model_data[metric2].dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 3:
            continue
        
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        results.append({
            'model': model,
            'n': len(x),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r_squared': pearson_r ** 2,
            'pearson_significant': pearson_p < 0.05,
            'spearman_significant': spearman_p < 0.05
        })
    
    return pd.DataFrame(results)


def calculate_per_culture_correlation(df, metric1, metric2):
    """Calculate correlations for each culture separately."""
    results = []
    
    for culture in df['culture'].unique():
        culture_data = df[df['culture'] == culture]
        x = culture_data[metric1].dropna()
        y = culture_data[metric2].dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 3:
            continue
        
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        results.append({
            'culture': culture,
            'n': len(x),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r_squared': pearson_r ** 2,
            'pearson_significant': pearson_p < 0.05
        })
    
    return pd.DataFrame(results)


def calculate_all_metric_correlations(df):
    """Calculate correlation matrix for all metrics."""
    metrics = get_all_metrics()
    
    # Pearson correlation matrix
    pearson_corr = df[metrics].corr(method='pearson')
    
    # Spearman correlation matrix
    spearman_corr = df[metrics].corr(method='spearman')
    
    # P-values matrix
    n = len(df)
    p_values = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    
    for m1 in metrics:
        for m2 in metrics:
            if m1 == m2:
                p_values.loc[m1, m2] = 0.0
            else:
                _, p = stats.pearsonr(df[m1].dropna(), df[m2].dropna())
                p_values.loc[m1, m2] = p
    
    return pearson_corr, spearman_corr, p_values


def interpret_correlation(r):
    """Interpret correlation coefficient strength."""
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.5:
        strength = "Moderate"
    elif abs_r >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak/Negligible"
    
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def check_hallucination_hypothesis(pearson_r):
    """
    Check if models are 'hallucinating fluently'.
    Hypothesis: If r < 0.3, models can produce fluent text without cultural accuracy.
    """
    if pearson_r < 0.3:
        return {
            'hallucinating': True,
            'interpretation': "EVIDENCE OF FLUENT HALLUCINATION: Models can generate linguistically fluent text without corresponding cultural accuracy.",
            'recommendation': "Cultural accuracy training/fine-tuning needed."
        }
    elif pearson_r < 0.5:
        return {
            'hallucinating': 'Partial',
            'interpretation': "MODERATE LINK: Some decoupling between fluency and accuracy exists.",
            'recommendation': "Additional cultural grounding may improve accuracy."
        }
    elif pearson_r < 0.7:
        return {
            'hallucinating': False,
            'interpretation': "MODERATE-STRONG LINK: Fluency and accuracy are reasonably coupled.",
            'recommendation': "Models show reasonable cultural awareness."
        }
    else:
        return {
            'hallucinating': False,
            'interpretation': "STRONG LINK: Fluency implies accuracy - models are culturally grounded.",
            'recommendation': "Models demonstrate strong cultural understanding."
        }


def perform_regression_analysis(df, x_col, y_col):
    """Perform linear regression analysis."""
    x = df[x_col].values
    y = df[y_col].values
    
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Simple linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    result = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'equation': f"y = {slope:.4f}x + {intercept:.4f}"
    }
    
    # OLS regression with statsmodels for more details
    if HAS_STATSMODELS:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        result['ols_r_squared'] = model.rsquared
        result['ols_adj_r_squared'] = model.rsquared_adj
        result['ols_f_statistic'] = model.fvalue
        result['ols_f_pvalue'] = model.f_pvalue
        result['confidence_intervals'] = model.conf_int().tolist()
    
    return result


def create_visualizations(df, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    metrics = get_key_metrics()
    linguistic = metrics['linguistic']
    cultural = metrics['cultural']
    all_metrics = get_all_metrics()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Main scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.regplot(
        data=df,
        x=linguistic,
        y=cultural,
        scatter_kws={'alpha': 0.5, 's': 50},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )
    
    # Calculate correlation for title
    r, p = stats.pearsonr(df[linguistic], df[cultural])
    
    ax.set_title(f'RQ4: Linguistic Fluency vs Cultural Accuracy\nPearson r = {r:.3f}, p = {p:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Linguistic & Expressive Appropriateness', fontsize=11)
    ax.set_ylabel('Cultural Accuracy & Authenticity', fontsize=11)
    
    # Add interpretation text
    interpretation = interpret_correlation(r)
    ax.text(0.05, 0.95, f'Correlation: {interpretation}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq4_scatter_linguistic_vs_cultural.png', bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot colored by model
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = df['model'].unique()
    palette = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax.scatter(model_data[linguistic], model_data[cultural], 
                   alpha=0.6, s=60, label=model, color=palette[i])
    
    # Add overall regression line
    slope, intercept, _, _, _ = stats.linregress(df[linguistic], df[cultural])
    x_line = np.array([df[linguistic].min(), df[linguistic].max()])
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label='Overall trend')
    
    ax.set_title('RQ4: Linguistic vs Cultural Accuracy by Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Linguistic & Expressive Appropriateness', fontsize=11)
    ax.set_ylabel('Cultural Accuracy & Authenticity', fontsize=11)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq4_scatter_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 3. Per-model regression lines
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        sns.regplot(
            data=model_data,
            x=linguistic,
            y=cultural,
            scatter=False,
            label=model,
            color=palette[i],
            ax=ax
        )
    
    ax.set_title('RQ4: Regression Lines by Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Linguistic & Expressive Appropriateness', fontsize=11)
    ax.set_ylabel('Cultural Accuracy & Authenticity', fontsize=11)
    ax.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq4_regression_lines_by_model.png', bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap for all metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[all_metrics].corr(method='pearson')
    
    # Shorten metric names for display
    short_names = ['Cultural', 'Contextual', 'Narrative', 'Linguistic']
    corr_display = corr_matrix.copy()
    corr_display.index = short_names
    corr_display.columns = short_names
    
    mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
    
    sns.heatmap(
        corr_display,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Pearson r'}
    )
    
    ax.set_title('RQ4: Correlation Matrix - All Evaluation Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq4_correlation_matrix.png', bbox_inches='tight')
    plt.close()
    
    # 5. Bar chart of per-model correlations
    model_corr = calculate_per_model_correlation(df, linguistic, cultural)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_corr))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_corr['pearson_r'], width, label='Pearson r', color='steelblue')
    bars2 = ax.bar(x + width/2, model_corr['spearman_r'], width, label='Spearman ρ', color='darkorange')
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('RQ4: Correlation Strength by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_corr['model'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Weak threshold (0.3)')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Strong threshold (0.7)')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq4_correlation_by_model.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, overall_corr, model_corr, hallucination_check, regression):
    """Generate summary of findings."""
    metrics = get_key_metrics()
    
    # Find model with strongest/weakest correlation
    model_corr_sorted = model_corr.sort_values('pearson_r', ascending=False)
    strongest_model = model_corr_sorted.iloc[0]
    weakest_model = model_corr_sorted.iloc[-1]
    
    summary = {
        'research_question': 'RQ4: Linguistic Fluency vs Cultural Accuracy Correlation',
        'main_question': 'Are models hallucinating fluently?',
        'total_stories': len(df),
        'num_models': df['model'].nunique(),
        
        # Overall correlation
        'overall_pearson_r': overall_corr['pearson_r'],
        'overall_pearson_p': overall_corr['pearson_p'],
        'overall_spearman_r': overall_corr['spearman_r'],
        'overall_spearman_p': overall_corr['spearman_p'],
        'overall_r_squared': overall_corr['r_squared'],
        'correlation_strength': interpret_correlation(overall_corr['pearson_r']),
        
        # Hallucination analysis
        'hallucinating_fluently': hallucination_check['hallucinating'],
        'hallucination_interpretation': hallucination_check['interpretation'],
        'recommendation': hallucination_check['recommendation'],
        
        # Regression
        'regression_equation': regression['equation'],
        'regression_slope': regression['slope'],
        'regression_intercept': regression['intercept'],
        
        # Per-model findings
        'strongest_correlation_model': strongest_model['model'],
        'strongest_correlation_r': strongest_model['pearson_r'],
        'weakest_correlation_model': weakest_model['model'],
        'weakest_correlation_r': weakest_model['pearson_r'],
        'correlation_range': strongest_model['pearson_r'] - weakest_model['pearson_r']
    }
    
    return summary


def main():
    """Main execution function for RQ4 analysis."""
    print("=" * 70)
    print("RQ4: Linguistic Fluency vs. Cultural Accuracy Correlation")
    print("=" * 70)
    print("\nQuestion: Are models 'hallucinating fluently'?")
    print("Method: Pearson/Spearman correlation + per-model analysis")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} stories from {df['model'].nunique()} models")
    
    metrics = get_key_metrics()
    linguistic = metrics['linguistic']
    cultural = metrics['cultural']
    
    # Overall correlation
    print("\n[2] Calculating overall correlation...")
    overall_corr = calculate_overall_correlation(df, linguistic, cultural)
    
    overall_corr_df = pd.DataFrame([{
        'metric_1': linguistic,
        'metric_2': cultural,
        **overall_corr
    }])
    overall_corr_df.to_csv(output_dir / 'rq4_overall_correlation.csv', index=False)
    print(f"    Exported: rq4_overall_correlation.csv")
    
    print(f"\n    Overall Correlation Results:")
    print(f"    {'='*50}")
    print(f"    Pearson r:  {overall_corr['pearson_r']:.4f} (p = {overall_corr['pearson_p']:.6f})")
    print(f"    Spearman ρ: {overall_corr['spearman_r']:.4f} (p = {overall_corr['spearman_p']:.6f})")
    print(f"    R-squared:  {overall_corr['r_squared']:.4f}")
    print(f"    Strength:   {interpret_correlation(overall_corr['pearson_r'])}")
    
    # Per-model correlation
    print("\n[3] Calculating per-model correlations...")
    model_corr = calculate_per_model_correlation(df, linguistic, cultural)
    model_corr = model_corr.sort_values('pearson_r', ascending=False)
    model_corr.to_csv(output_dir / 'rq4_correlation_by_model.csv', index=False)
    print(f"    Exported: rq4_correlation_by_model.csv")
    
    print("\n    Per-Model Correlation Results:")
    print("    " + "-" * 70)
    print(f"    {'Model':<40} {'Pearson r':>10} {'p-value':>12} {'Sig?':>8}")
    print("    " + "-" * 70)
    for _, row in model_corr.iterrows():
        sig = '*' if row['pearson_significant'] else ''
        print(f"    {row['model']:<40} {row['pearson_r']:>10.4f} {row['pearson_p']:>12.6f} {sig:>8}")
    
    # Per-culture correlation
    print("\n[4] Calculating per-culture correlations...")
    culture_corr = calculate_per_culture_correlation(df, linguistic, cultural)
    culture_corr = culture_corr.sort_values('pearson_r', ascending=False)
    culture_corr.to_csv(output_dir / 'rq4_correlation_by_culture.csv', index=False)
    print(f"    Exported: rq4_correlation_by_culture.csv")
    
    # All metrics correlation matrix
    print("\n[5] Calculating full correlation matrix...")
    pearson_corr, spearman_corr, p_values = calculate_all_metric_correlations(df)
    
    pearson_corr.to_csv(output_dir / 'rq4_correlation_matrix_pearson.csv')
    spearman_corr.to_csv(output_dir / 'rq4_correlation_matrix_spearman.csv')
    p_values.to_csv(output_dir / 'rq4_correlation_pvalues.csv')
    print(f"    Exported: rq4_correlation_matrix_pearson.csv")
    print(f"    Exported: rq4_correlation_matrix_spearman.csv")
    print(f"    Exported: rq4_correlation_pvalues.csv")
    
    # Regression analysis
    print("\n[6] Performing regression analysis...")
    regression = perform_regression_analysis(df, linguistic, cultural)
    
    regression_df = pd.DataFrame([regression])
    regression_df.to_csv(output_dir / 'rq4_regression_analysis.csv', index=False)
    print(f"    Exported: rq4_regression_analysis.csv")
    
    print(f"\n    Regression Results:")
    print(f"    Equation: {regression['equation']}")
    print(f"    R-squared: {regression['r_squared']:.4f}")
    print(f"    p-value: {regression['p_value']:.6f}")
    
    # Hallucination hypothesis check
    print("\n[7] Checking hallucination hypothesis...")
    hallucination_check = check_hallucination_hypothesis(overall_corr['pearson_r'])
    
    hallucination_df = pd.DataFrame([hallucination_check])
    hallucination_df.to_csv(output_dir / 'rq4_hallucination_analysis.csv', index=False)
    print(f"    Exported: rq4_hallucination_analysis.csv")
    
    # Create visualizations
    print("\n[8] Creating visualizations...")
    create_visualizations(df, output_dir)
    
    # Generate summary
    print("\n[9] Generating summary...")
    summary = generate_summary(df, overall_corr, model_corr, hallucination_check, regression)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq4_summary.csv', index=False)
    print(f"    Exported: rq4_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ4 ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nKEY FINDING: {hallucination_check['interpretation']}")
    print(f"\n  Overall Pearson r: {overall_corr['pearson_r']:.4f}")
    print(f"  Correlation Strength: {interpret_correlation(overall_corr['pearson_r'])}")
    
    if overall_corr['pearson_r'] < 0.3:
        print("\n  ⚠️  WARNING: Weak correlation suggests models CAN hallucinate fluently!")
        print("     High linguistic scores do not guarantee cultural accuracy.")
    elif overall_corr['pearson_r'] < 0.5:
        print("\n  ⚡ MODERATE: Some decoupling between fluency and accuracy.")
    else:
        print("\n  ✓  GOOD: Fluency and accuracy are reasonably linked.")
    
    print(f"\n  Strongest correlation: {model_corr.iloc[0]['model']} (r = {model_corr.iloc[0]['pearson_r']:.4f})")
    print(f"  Weakest correlation:  {model_corr.iloc[-1]['model']} (r = {model_corr.iloc[-1]['pearson_r']:.4f})")
    
    print(f"\n  Recommendation: {hallucination_check['recommendation']}")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq4_overall_correlation.csv - Main correlation results")
    print("  2. rq4_correlation_by_model.csv - Per-model correlations")
    print("  3. rq4_correlation_by_culture.csv - Per-culture correlations")
    print("  4. rq4_correlation_matrix_pearson.csv - Full Pearson matrix")
    print("  5. rq4_correlation_matrix_spearman.csv - Full Spearman matrix")
    print("  6. rq4_correlation_pvalues.csv - P-values matrix")
    print("  7. rq4_regression_analysis.csv - Regression results")
    print("  8. rq4_hallucination_analysis.csv - Hallucination hypothesis")
    print("  9. rq4_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, overall_corr, model_corr, hallucination_check


if __name__ == "__main__":
    df, overall_corr, model_corr, hallucination_check = main()
