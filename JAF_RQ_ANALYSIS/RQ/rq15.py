"""
RQ15: Best Embedding Model for Automated Evaluation
====================================================
Question: Which embedding correlates best with human judgment?
          Which embedding model is most useful for predicting cultural quality scores?

Method:
- For each of 5 parquet files: Repeat RQ11 regression analysis
- Compare R² scores across all 5 embedding models
- Per-metric analysis: Which embedding best predicts which metric?
- Recommend: Best embedding model for future auto-evaluation

Data Source: All 5 Parquet files + CSV (evaluation scores)

Embedding Models:
1. openai_text-embedding-3-large
2. openai_text-embedding-3-small
3. google_gemini-embedding-001
4. mistralai_mistral-embed-2312
5. qwen_qwen3-embedding-8b
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

# Embedding models
EMBEDDING_MODELS = [
    'openai_text-embedding-3-large',
    'openai_text-embedding-3-small',
    'google_gemini-embedding-001',
    'mistralai_mistral-embed-2312',
    'qwen_qwen3-embedding-8b'
]

# Short names for embedding models
EMBEDDING_SHORT_NAMES = {
    'openai_text-embedding-3-large': 'OpenAI-Large',
    'openai_text-embedding-3-small': 'OpenAI-Small',
    'google_gemini-embedding-001': 'Gemini',
    'mistralai_mistral-embed-2312': 'Mistral',
    'qwen_qwen3-embedding-8b': 'Qwen'
}


def load_csv_data():
    """Load CSV with evaluation scores."""
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "evaluated_stories.csv"
    return pd.read_csv(csv_path)


def load_parquet_data(embedding_model):
    """Load parquet with specific embedding model."""
    data_dir = Path(__file__).parent.parent / "data" / "parquet"
    parquet_path = data_dir / f"{embedding_model}.parquet"
    
    if not parquet_path.exists():
        print(f"    Warning: {parquet_path} not found")
        return None
    
    return pd.read_parquet(parquet_path)


def merge_data(csv_df, parquet_df):
    """Merge CSV with parquet data."""
    merged = csv_df.merge(
        parquet_df[['model', 'culture', 'story', 'story_embedding']],
        on=['model', 'culture', 'story'],
        how='inner'
    )
    return merged


def train_prediction_models(df, metric, random_state=42):
    """
    Train multiple regression models to predict a metric from embeddings.
    Returns performance metrics for each model type.
    """
    # Prepare features
    X = np.array(df['story_embedding'].tolist())
    y = df[metric].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to try (removed GradientBoosting for speed with high-dim embeddings)
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=random_state),
        'Lasso': Lasso(alpha=0.01, random_state=random_state, max_iter=5000),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, 
                                              random_state=random_state, n_jobs=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train
        if model_name in ['Ridge', 'Lasso']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # Cross-validation (3-fold for speed)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[model_name] = {
            'r2': r2,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    return results


def evaluate_embedding_model(embedding_model, csv_df):
    """
    Evaluate a single embedding model's predictive power for all metrics.
    """
    print(f"\n  Processing: {EMBEDDING_SHORT_NAMES.get(embedding_model, embedding_model)}")
    
    # Load parquet
    parquet_df = load_parquet_data(embedding_model)
    if parquet_df is None:
        return None
    
    # Merge
    merged = merge_data(csv_df, parquet_df)
    print(f"    Merged data: {len(merged)} stories")
    
    if len(merged) < 100:
        print(f"    Warning: Too few samples for reliable evaluation")
        return None
    
    # Get embedding dimension
    emb_dim = len(merged['story_embedding'].iloc[0])
    print(f"    Embedding dimension: {emb_dim}")
    
    results = []
    
    for metric in TARGET_METRICS:
        short_name = METRIC_SHORT_NAMES[metric]
        
        # Train models
        model_results = train_prediction_models(merged, metric)
        
        # Find best model
        best_model = max(model_results.items(), key=lambda x: x[1]['r2'])
        
        for model_name, metrics in model_results.items():
            results.append({
                'embedding_model': embedding_model,
                'embedding_short': EMBEDDING_SHORT_NAMES.get(embedding_model, embedding_model),
                'embedding_dimension': emb_dim,
                'target_metric': metric,
                'metric_short': short_name,
                'regression_model': model_name,
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'cv_r2_mean': metrics['cv_r2_mean'],
                'cv_r2_std': metrics['cv_r2_std'],
                'is_best_model': model_name == best_model[0]
            })
    
    return pd.DataFrame(results)


def run_all_embedding_evaluations(csv_df):
    """Run evaluation for all embedding models."""
    all_results = []
    
    for emb_model in EMBEDDING_MODELS:
        results = evaluate_embedding_model(emb_model, csv_df)
        if results is not None:
            all_results.append(results)
    
    if not all_results:
        raise ValueError("No embedding models could be evaluated")
    
    return pd.concat(all_results, ignore_index=True)


def create_comparison_summary(all_results):
    """Create summary comparison across embedding models."""
    # Best R² per embedding model per metric (using best regression model)
    best_per_metric = all_results[all_results['is_best_model']].copy()
    
    # Pivot: embedding × metric
    pivot = best_per_metric.pivot_table(
        values='r2',
        index='embedding_short',
        columns='metric_short',
        aggfunc='first'
    )
    
    # Add average R²
    pivot['average_r2'] = pivot.mean(axis=1)
    
    # Rank embeddings
    pivot['rank'] = pivot['average_r2'].rank(ascending=False)
    
    return pivot.sort_values('average_r2', ascending=False)


def create_best_embedding_per_metric(all_results):
    """Find best embedding for each target metric."""
    best_per_metric = all_results[all_results['is_best_model']].copy()
    
    results = []
    for metric in TARGET_METRICS:
        short_name = METRIC_SHORT_NAMES[metric]
        metric_data = best_per_metric[best_per_metric['metric_short'] == short_name]
        
        best_idx = metric_data['r2'].idxmax()
        best_row = metric_data.loc[best_idx]
        
        results.append({
            'target_metric': metric,
            'metric_short': short_name,
            'best_embedding': best_row['embedding_model'],
            'best_embedding_short': best_row['embedding_short'],
            'best_r2': best_row['r2'],
            'best_regression_model': best_row['regression_model'],
            'embedding_dimension': best_row['embedding_dimension']
        })
    
    # Also add average score prediction
    avg_scores = best_per_metric.groupby('embedding_short')['r2'].mean()
    best_avg = avg_scores.idxmax()
    
    results.append({
        'target_metric': 'Average across all metrics',
        'metric_short': 'average',
        'best_embedding': [k for k, v in EMBEDDING_SHORT_NAMES.items() if v == best_avg][0],
        'best_embedding_short': best_avg,
        'best_r2': avg_scores[best_avg],
        'best_regression_model': 'Mixed',
        'embedding_dimension': 'N/A'
    })
    
    return pd.DataFrame(results)


def create_embedding_ranking(all_results):
    """Create overall ranking of embedding models."""
    best_per_metric = all_results[all_results['is_best_model']].copy()
    
    ranking = best_per_metric.groupby(['embedding_model', 'embedding_short', 'embedding_dimension']).agg({
        'r2': ['mean', 'std', 'min', 'max'],
        'cv_r2_mean': 'mean'
    }).reset_index()
    
    ranking.columns = ['embedding_model', 'embedding_short', 'embedding_dimension',
                      'mean_r2', 'std_r2', 'min_r2', 'max_r2', 'mean_cv_r2']
    
    ranking['rank'] = ranking['mean_r2'].rank(ascending=False)
    ranking = ranking.sort_values('rank')
    
    # Count how many metrics this embedding is best for
    best_counts = best_per_metric.loc[
        best_per_metric.groupby('metric_short')['r2'].idxmax()
    ]['embedding_short'].value_counts()
    
    ranking['metrics_best_for'] = ranking['embedding_short'].map(best_counts).fillna(0).astype(int)
    
    return ranking


def compute_statistical_significance(all_results):
    """Test if differences between embeddings are statistically significant."""
    best_per_metric = all_results[all_results['is_best_model']].copy()
    
    results = []
    embeddings = best_per_metric['embedding_short'].unique()
    
    for i, emb1 in enumerate(embeddings):
        for emb2 in embeddings[i+1:]:
            scores1 = best_per_metric[best_per_metric['embedding_short'] == emb1]['r2'].values
            scores2 = best_per_metric[best_per_metric['embedding_short'] == emb2]['r2'].values
            
            # Paired t-test (same metrics)
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            results.append({
                'embedding_1': emb1,
                'embedding_2': emb2,
                'mean_r2_1': scores1.mean(),
                'mean_r2_2': scores2.mean(),
                'difference': scores1.mean() - scores2.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)


def analyze_by_regression_model(all_results):
    """Analyze which regression model works best across embeddings."""
    summary = all_results.groupby(['regression_model', 'embedding_short']).agg({
        'r2': 'mean',
        'cv_r2_mean': 'mean'
    }).reset_index()
    
    # Overall regression model ranking
    overall = all_results.groupby('regression_model').agg({
        'r2': ['mean', 'std'],
        'cv_r2_mean': 'mean'
    }).reset_index()
    overall.columns = ['regression_model', 'mean_r2', 'std_r2', 'mean_cv_r2']
    overall['rank'] = overall['mean_r2'].rank(ascending=False)
    overall = overall.sort_values('rank')
    
    return summary, overall


def create_visualizations(all_results, comparison, ranking, best_per_metric, 
                         regression_analysis, output_dir):
    """Create all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Heatmap: Embedding × Metric R² scores
    fig, ax = plt.subplots(figsize=(12, 8))
    
    best_per_embed = all_results[all_results['is_best_model']].copy()
    pivot = best_per_embed.pivot_table(
        values='r2',
        index='embedding_short',
        columns='metric_short',
        aggfunc='first'
    )
    
    # Reorder by average R²
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'R² Score'}, vmin=0, vmax=0.3)
    ax.set_title('RQ15: Embedding Model Performance (R²) by Target Metric', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Metric', fontsize=11)
    ax.set_ylabel('Embedding Model', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_heatmap_embedding_metric.png', bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart: Overall embedding ranking
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ranking_sorted = ranking.sort_values('mean_r2', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ranking_sorted)))[::-1]
    
    bars = ax.bar(range(len(ranking_sorted)), ranking_sorted['mean_r2'], 
                  yerr=ranking_sorted['std_r2'], capsize=5,
                  color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(ranking_sorted)))
    ax.set_xticklabels(ranking_sorted['embedding_short'], rotation=45, ha='right')
    ax.set_ylabel('Mean R² Score', fontsize=11)
    ax.set_title('RQ15: Embedding Model Ranking by Predictive Power', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, ranking_sorted['mean_r2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_embedding_ranking.png', bbox_inches='tight')
    plt.close()
    
    # 3. Grouped bar: Performance by metric
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metrics = list(METRIC_SHORT_NAMES.values())
    embeddings = list(EMBEDDING_SHORT_NAMES.values())
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, emb in enumerate(embeddings):
        emb_data = best_per_embed[best_per_embed['embedding_short'] == emb]
        values = [emb_data[emb_data['metric_short'] == m]['r2'].values[0] if len(emb_data[emb_data['metric_short'] == m]) > 0 else 0 for m in metrics]
        ax.bar(x + i*width, values, width, label=emb)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('RQ15: Embedding Performance by Target Metric', fontsize=14, fontweight='bold')
    ax.legend(title='Embedding Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_grouped_by_metric.png', bbox_inches='tight')
    plt.close()
    
    # 4. Regression model comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    reg_summary, reg_overall = regression_analysis
    reg_overall_sorted = reg_overall.sort_values('mean_r2', ascending=False)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(reg_overall_sorted)))
    bars = ax.bar(range(len(reg_overall_sorted)), reg_overall_sorted['mean_r2'],
                  yerr=reg_overall_sorted['std_r2'], capsize=5,
                  color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(reg_overall_sorted)))
    ax.set_xticklabels(reg_overall_sorted['regression_model'], rotation=45, ha='right')
    ax.set_ylabel('Mean R² Score', fontsize=11)
    ax.set_title('RQ15: Regression Model Performance Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_regression_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 5. Best embedding per metric
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_df = best_per_metric[best_per_metric['metric_short'] != 'average']
    colors_map = {emb: plt.cm.tab10(i) for i, emb in enumerate(embeddings)}
    bar_colors = [colors_map.get(e, 'gray') for e in best_df['best_embedding_short']]
    
    bars = ax.bar(range(len(best_df)), best_df['best_r2'], color=bar_colors, edgecolor='black')
    
    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels(best_df['metric_short'], rotation=45, ha='right')
    ax.set_ylabel('Best R² Score', fontsize=11)
    ax.set_title('RQ15: Best Performing Embedding for Each Metric', fontsize=14, fontweight='bold')
    
    # Add embedding name labels
    for bar, emb in zip(bars, best_df['best_embedding_short']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               emb, ha='center', fontsize=9, rotation=90, va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_best_per_metric.png', bbox_inches='tight')
    plt.close()
    
    # 6. Dimension vs Performance scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter_data = ranking.copy()
    scatter_data['embedding_dimension'] = scatter_data['embedding_dimension'].astype(int)
    
    ax.scatter(scatter_data['embedding_dimension'], scatter_data['mean_r2'], 
               s=150, c=range(len(scatter_data)), cmap='viridis', edgecolors='black')
    
    for idx, row in scatter_data.iterrows():
        ax.annotate(row['embedding_short'], 
                   (row['embedding_dimension'], row['mean_r2']),
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel('Embedding Dimension', fontsize=11)
    ax.set_ylabel('Mean R² Score', fontsize=11)
    ax.set_title('RQ15: Embedding Dimension vs. Predictive Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq15_dimension_vs_performance.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_recommendations(ranking, best_per_metric, significance):
    """Generate recommendations based on analysis."""
    best_overall = ranking.iloc[0]
    
    recommendations = {
        'best_overall_embedding': best_overall['embedding_model'],
        'best_overall_embedding_short': best_overall['embedding_short'],
        'best_overall_r2': best_overall['mean_r2'],
        'best_overall_dimension': best_overall['embedding_dimension'],
        
        'recommendation': f"Use {best_overall['embedding_short']} for automated cultural evaluation",
        'reasoning': f"Achieves highest mean R² of {best_overall['mean_r2']:.4f} across all metrics",
        
        # Per-metric recommendations
        'best_for_cultural': best_per_metric[best_per_metric['metric_short'] == 'cultural_accuracy']['best_embedding_short'].values[0],
        'best_for_contextual': best_per_metric[best_per_metric['metric_short'] == 'contextual_appropriateness']['best_embedding_short'].values[0],
        'best_for_narrative': best_per_metric[best_per_metric['metric_short'] == 'narrative_coherence']['best_embedding_short'].values[0],
        'best_for_linguistic': best_per_metric[best_per_metric['metric_short'] == 'linguistic_fluency']['best_embedding_short'].values[0]
    }
    
    # Check if differences are significant
    if len(significance) > 0:
        top_comparisons = significance[significance['embedding_1'] == best_overall['embedding_short']]
        all_significant = top_comparisons['significant'].all() if len(top_comparisons) > 0 else False
        recommendations['significantly_better'] = all_significant
    else:
        recommendations['significantly_better'] = False
    
    return recommendations


def main():
    """Main execution function for RQ15 analysis."""
    print("=" * 70)
    print("RQ15: Best Embedding Model for Automated Evaluation")
    print("=" * 70)
    print("\nQuestion: Which embedding correlates best with human judgment?")
    print("Method: Compare predictive power of 5 embedding models")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load CSV data
    print("\n[1] Loading evaluation data...")
    csv_df = load_csv_data()
    print(f"    Loaded {len(csv_df)} stories with human scores")
    
    # Evaluate all embedding models
    print("\n[2] Evaluating embedding models...")
    print("    Testing: Ridge, Lasso, RandomForest, GradientBoosting")
    all_results = run_all_embedding_evaluations(csv_df)
    all_results.to_csv(output_dir / 'rq15_all_results.csv', index=False)
    print(f"\n    Exported: rq15_all_results.csv ({len(all_results)} rows)")
    
    # Create comparison summary
    print("\n[3] Creating comparison summary...")
    comparison = create_comparison_summary(all_results)
    comparison.to_csv(output_dir / 'rq15_comparison_matrix.csv')
    print(f"    Exported: rq15_comparison_matrix.csv")
    
    print("\n    R² Comparison Matrix (best regression model per cell):")
    print(comparison.to_string())
    
    # Best embedding per metric
    print("\n[4] Finding best embedding for each metric...")
    best_per_metric = create_best_embedding_per_metric(all_results)
    best_per_metric.to_csv(output_dir / 'rq15_best_per_metric.csv', index=False)
    print(f"    Exported: rq15_best_per_metric.csv")
    
    print("\n    Best Embedding per Metric:")
    for _, row in best_per_metric.iterrows():
        print(f"      {row['metric_short']}: {row['best_embedding_short']} (R²={row['best_r2']:.4f})")
    
    # Create embedding ranking
    print("\n[5] Creating embedding ranking...")
    ranking = create_embedding_ranking(all_results)
    ranking.to_csv(output_dir / 'rq15_embedding_ranking.csv', index=False)
    print(f"    Exported: rq15_embedding_ranking.csv")
    
    print("\n    Overall Embedding Ranking:")
    for _, row in ranking.iterrows():
        print(f"      #{int(row['rank'])} {row['embedding_short']}: Mean R²={row['mean_r2']:.4f} (dim={row['embedding_dimension']})")
    
    # Statistical significance
    print("\n[6] Testing statistical significance...")
    significance = compute_statistical_significance(all_results)
    significance.to_csv(output_dir / 'rq15_significance_tests.csv', index=False)
    print(f"    Exported: rq15_significance_tests.csv")
    
    sig_count = significance['significant'].sum()
    print(f"\n    Significant differences: {sig_count}/{len(significance)} pairs")
    
    # Regression model analysis
    print("\n[7] Analyzing regression models...")
    regression_analysis = analyze_by_regression_model(all_results)
    reg_summary, reg_overall = regression_analysis
    reg_summary.to_csv(output_dir / 'rq15_regression_by_embedding.csv', index=False)
    reg_overall.to_csv(output_dir / 'rq15_regression_overall.csv', index=False)
    print(f"    Exported: rq15_regression_by_embedding.csv")
    print(f"    Exported: rq15_regression_overall.csv")
    
    print("\n    Regression Model Ranking:")
    for _, row in reg_overall.iterrows():
        print(f"      #{int(row['rank'])} {row['regression_model']}: Mean R²={row['mean_r2']:.4f}")
    
    # Generate recommendations
    print("\n[8] Generating recommendations...")
    recommendations = generate_recommendations(ranking, best_per_metric, significance)
    rec_df = pd.DataFrame([recommendations])
    rec_df.to_csv(output_dir / 'rq15_recommendations.csv', index=False)
    print(f"    Exported: rq15_recommendations.csv")
    
    # Create visualizations
    print("\n[9] Creating visualizations...")
    create_visualizations(all_results, comparison, ranking, best_per_metric,
                         regression_analysis, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ15 ANALYSIS COMPLETE")
    print("=" * 70)
    
    best = ranking.iloc[0]
    worst = ranking.iloc[-1]
    
    print(f"\nKey Findings:")
    
    print(f"\n  Best Embedding Model for Automated Evaluation:")
    print(f"    🥇 {best['embedding_short']} (dim={best['embedding_dimension']})")
    print(f"       Mean R² = {best['mean_r2']:.4f}")
    print(f"       Best for {best['metrics_best_for']} metric(s)")
    
    print(f"\n  Worst Embedding Model:")
    print(f"    🥉 {worst['embedding_short']} (dim={worst['embedding_dimension']})")
    print(f"       Mean R² = {worst['mean_r2']:.4f}")
    
    improvement = (best['mean_r2'] - worst['mean_r2']) / worst['mean_r2'] * 100 if worst['mean_r2'] > 0 else 0
    print(f"\n  Performance Gap:")
    print(f"    {improvement:.1f}% improvement from worst to best embedding")
    
    print(f"\n  Best Embedding per Metric:")
    for _, row in best_per_metric[best_per_metric['metric_short'] != 'average'].iterrows():
        print(f"    • {row['metric_short']}: {row['best_embedding_short']}")
    
    print(f"\n  Recommendation:")
    print(f"    → {recommendations['recommendation']}")
    print(f"    → {recommendations['reasoning']}")
    
    # Caveat about R² values
    if best['mean_r2'] < 0.2:
        print(f"\n  ⚠️  Note: All R² values are relatively low (<0.2)")
        print(f"     Embeddings alone have limited predictive power for human scores")
        print(f"     Consider combining with other features for better prediction")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq15_all_results.csv - All regression results")
    print("  2. rq15_comparison_matrix.csv - Embedding × Metric R² matrix")
    print("  3. rq15_best_per_metric.csv - Best embedding for each metric")
    print("  4. rq15_embedding_ranking.csv - Overall embedding ranking")
    print("  5. rq15_significance_tests.csv - Statistical significance tests")
    print("  6. rq15_regression_by_embedding.csv - Regression model breakdown")
    print("  7. rq15_regression_overall.csv - Overall regression ranking")
    print("  8. rq15_recommendations.csv - Final recommendations")
    print("-" * 70)
    
    return all_results, ranking, recommendations


if __name__ == "__main__":
    all_results, ranking, recommendations = main()
