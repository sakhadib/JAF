"""
RQ11: Automated Score Prediction ("AI Judge")
=============================================
Question: Can embeddings predict human evaluation scores?
          Can we build an automated system to predict story quality from embeddings?

Method:
- Join embeddings (X) with human evaluation scores (Y)
- Train-test split: 80/20 stratified by culture
- Train predictive models: Ridge Regression, Random Forest
- Evaluate: R², MAE, RMSE for each of 4 metrics
- Feature importance: Which embedding dimensions matter most?

Data Source: Parquet (story_embedding) + CSV (evaluation scores)

Interpretation:
- High R² → Embeddings capture quality-related features
- Low R² → Human judgment based on factors not in embeddings
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


# Target metrics to predict (actual column names from CSV)
TARGET_METRICS = [
    'Cultural Accuracy & Authenticity',
    'Contextual & Temporal Appropriateness',
    'Narrative & Symbolic Coherence',
    'Linguistic & Expressive Appropriateness (Bangla)'
]

# Shorter display names for outputs
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
    
    # Merge on common columns: model, culture, story
    # The parquet has: model, culture, story, story_type, region, story_embedding, scenario_embedding
    # The CSV has: story_id, culture, model, story + score columns
    
    merged = csv_df.merge(
        parquet_df[['model', 'culture', 'story', 'story_type', 'region', 'story_embedding']],
        on=['model', 'culture', 'story'],
        how='inner'
    )
    
    return merged


def prepare_features_and_targets(df):
    """Prepare X (embeddings) and Y (scores) for modeling."""
    # Extract embeddings as feature matrix
    X = np.array(df['story_embedding'].tolist())
    
    # Extract targets
    Y = df[TARGET_METRICS].values
    
    # Metadata for stratification
    metadata = df[['model', 'culture', 'story_type', 'region']].copy()
    
    return X, Y, metadata, df


def train_and_evaluate_models(X_train, X_test, y_train, y_test, metric_name):
    """Train multiple models and evaluate performance."""
    results = []
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to try
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.01, random_state=42, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_r2 = -np.inf
    best_model_name = None
    
    for model_name, model in models.items():
        # Use scaled data for linear models
        if 'Forest' in model_name or 'Boosting' in model_name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation R² on training set
        if 'Forest' in model_name or 'Boosting' in model_name:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results.append({
            'target_metric': metric_name,
            'model': model_name,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        })
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = model_name
    
    return pd.DataFrame(results), best_model, best_model_name, scaler


def extract_feature_importance(model, model_name, n_features, top_n=50):
    """Extract feature importance from the best model."""
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_)
    
    if importance is not None:
        # Get top N important dimensions
        top_indices = np.argsort(importance)[-top_n:][::-1]
        
        importance_df = pd.DataFrame({
            'dimension': top_indices,
            'importance': importance[top_indices]
        })
        
        return importance_df
    
    return None


def perform_cross_model_analysis(X, Y, metadata):
    """Analyze how well embeddings predict scores across different models/cultures."""
    results = []
    
    le = LabelEncoder()
    
    for groupby_col in ['model', 'culture']:
        groups = metadata[groupby_col].unique()
        
        for group in groups:
            mask = metadata[groupby_col] == group
            X_group = X[mask]
            
            if len(X_group) < 10:
                continue
            
            for i, metric in enumerate(TARGET_METRICS):
                y_group = Y[mask, i]
                
                # Simple train-test split for this group
                if len(X_group) >= 20:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_group, y_group, test_size=0.2, random_state=42
                    )
                    
                    model = Ridge(alpha=1.0)
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    X_te_scaled = scaler.transform(X_te)
                    
                    model.fit(X_tr_scaled, y_tr)
                    y_pred = model.predict(X_te_scaled)
                    
                    r2 = r2_score(y_te, y_pred)
                    mae = mean_absolute_error(y_te, y_pred)
                else:
                    # Too few samples - use cross-validation only
                    model = Ridge(alpha=1.0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_group)
                    
                    cv_scores = cross_val_score(model, X_scaled, y_group, cv=min(5, len(X_group)//2), scoring='r2')
                    r2 = cv_scores.mean()
                    mae = np.nan
                
                results.append({
                    'groupby': groupby_col,
                    'group': group,
                    'target_metric': metric,
                    'r2_score': r2,
                    'mae': mae,
                    'n_samples': len(X_group)
                })
    
    return pd.DataFrame(results)


def compute_embedding_score_correlation(X, Y):
    """Compute correlation between embedding dimensions and scores."""
    correlations = []
    
    for i, metric in enumerate(TARGET_METRICS):
        y = Y[:, i]
        
        for dim in range(X.shape[1]):
            r, p = stats.pearsonr(X[:, dim], y)
            correlations.append({
                'target_metric': metric,
                'dimension': dim,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05
            })
    
    return pd.DataFrame(correlations)


def create_visualizations(all_results, cross_analysis, correlations, importance_dfs, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Model performance comparison (R² by metric and model type)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(TARGET_METRICS):
        metric_data = all_results[all_results['target_metric'] == metric]
        
        ax = axes[idx]
        bars = ax.bar(range(len(metric_data)), metric_data['r2_score'], 
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_data))),
                      edgecolor='black')
        
        ax.set_xticks(range(len(metric_data)))
        ax.set_xticklabels(metric_data['model'], rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        short_name = METRIC_SHORT_NAMES.get(metric, metric)
        ax.set_title(f'{short_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, metric_data['r2_score']):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('RQ11: Prediction Performance by Model Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq11_model_performance.png', bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: R² by target metric and ML model
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create pivot with short names for display
    all_results_display = all_results.copy()
    all_results_display['target_metric_short'] = all_results_display['target_metric'].map(METRIC_SHORT_NAMES)
    
    pivot = all_results_display.pivot_table(
        values='r2_score',
        index='model',
        columns='target_metric_short',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'label': 'R² Score'}, vmin=-0.5, vmax=0.5)
    ax.set_title('RQ11: R² Scores by Model and Target Metric', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq11_r2_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 3. Cross-analysis by generation model
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_analysis = cross_analysis[cross_analysis['groupby'] == 'model']
    model_analysis = model_analysis.copy()
    model_analysis['target_metric_short'] = model_analysis['target_metric'].map(METRIC_SHORT_NAMES)
    
    pivot = model_analysis.pivot_table(
        values='r2_score',
        index='group',
        columns='target_metric_short',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'label': 'R² Score'})
    ax.set_title('RQ11: Prediction Performance by Generation Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Metric')
    ax.set_ylabel('Generation Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq11_by_generation_model.png', bbox_inches='tight')
    plt.close()
    
    # 4. Cross-analysis by culture
    fig, ax = plt.subplots(figsize=(14, 10))
    
    culture_analysis = cross_analysis[cross_analysis['groupby'] == 'culture']
    culture_analysis = culture_analysis.copy()
    culture_analysis['target_metric_short'] = culture_analysis['target_metric'].map(METRIC_SHORT_NAMES)
    
    pivot = culture_analysis.pivot_table(
        values='r2_score',
        index='group',
        columns='target_metric_short',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'label': 'R² Score'})
    ax.set_title('RQ11: Prediction Performance by Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Metric')
    ax.set_ylabel('Culture')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq11_by_culture.png', bbox_inches='tight')
    plt.close()
    
    # 5. Feature importance (if available)
    if importance_dfs:
        n_metrics = len(importance_dfs)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (metric, imp_df) in enumerate(importance_dfs.items()):
            if imp_df is not None and idx < 4:
                ax = axes[idx]
                top_20 = imp_df.head(20)
                
                ax.barh(range(len(top_20)), top_20['importance'], color='steelblue')
                ax.set_yticks(range(len(top_20)))
                ax.set_yticklabels([f'Dim {d}' for d in top_20['dimension']])
                ax.set_xlabel('Importance')
                short_name = METRIC_SHORT_NAMES.get(metric, metric)
                ax.set_title(f'{short_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
                ax.invert_yaxis()
        
        plt.suptitle('RQ11: Top 20 Important Embedding Dimensions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'rq11_feature_importance.png', bbox_inches='tight')
        plt.close()
    
    # 6. Correlation distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(TARGET_METRICS):
        ax = axes[idx]
        metric_corr = correlations[correlations['target_metric'] == metric]
        
        ax.hist(metric_corr['correlation'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', label='Zero')
        ax.axvline(x=metric_corr['correlation'].mean(), color='orange', linestyle='-', 
                   label=f'Mean: {metric_corr["correlation"].mean():.3f}')
        
        n_sig = metric_corr['significant'].sum()
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Count')
        short_name = METRIC_SHORT_NAMES.get(metric, metric)
        ax.set_title(f'{short_name.replace("_", " ").title()}\n({n_sig} significant dims)', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.suptitle('RQ11: Embedding-Score Correlation Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rq11_correlation_distribution.png', bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(all_results, cross_analysis, correlations):
    """Generate summary of key findings."""
    # Best overall model per metric
    best_models = all_results.loc[all_results.groupby('target_metric')['r2_score'].idxmax()]
    
    # Average R² across all metrics and models
    avg_r2 = all_results['r2_score'].mean()
    max_r2 = all_results['r2_score'].max()
    
    # Best predicted metric
    best_metric = all_results.groupby('target_metric')['r2_score'].max().idxmax()
    best_metric_r2 = all_results.groupby('target_metric')['r2_score'].max().max()
    
    # Worst predicted metric
    worst_metric = all_results.groupby('target_metric')['r2_score'].max().idxmin()
    worst_metric_r2 = all_results.groupby('target_metric')['r2_score'].max().min()
    
    # Significant correlations
    n_significant = correlations['significant'].sum()
    total_correlations = len(correlations)
    
    summary = {
        'research_question': 'RQ11: Automated Score Prediction (AI Judge)',
        'main_question': 'Can embeddings predict human evaluation scores?',
        'n_samples': len(correlations) // len(TARGET_METRICS) // correlations['dimension'].max(),
        
        # Overall performance
        'average_r2_all': avg_r2,
        'max_r2_achieved': max_r2,
        
        # Best/worst metrics
        'best_predicted_metric': best_metric,
        'best_metric_r2': best_metric_r2,
        'worst_predicted_metric': worst_metric,
        'worst_metric_r2': worst_metric_r2,
        
        # Correlations
        'n_significant_correlations': n_significant,
        'total_correlations': total_correlations,
        'pct_significant': n_significant / total_correlations * 100,
        
        # Interpretation
        'embeddings_predictive': max_r2 > 0.1,
        'interpretation': 'Embeddings capture some quality signals' if max_r2 > 0.1 else 'Limited predictive power'
    }
    
    return summary


def main():
    """Main execution function for RQ11 analysis."""
    print("=" * 70)
    print("RQ11: Automated Score Prediction ('AI Judge')")
    print("=" * 70)
    print("\nQuestion: Can embeddings predict human evaluation scores?")
    print("Method: Train ML models on embeddings to predict quality metrics")
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
    print(f"    Merged dataset: {len(df)} stories")
    
    # Prepare features and targets
    print("\n[2] Preparing features and targets...")
    X, Y, metadata, full_df = prepare_features_and_targets(df)
    print(f"    Features shape: {X.shape}")
    print(f"    Targets: {TARGET_METRICS}")
    
    # Stratified split
    print("\n[3] Creating train-test split (80/20, stratified by culture)...")
    le = LabelEncoder()
    strat_labels = le.fit_transform(metadata['culture'])
    
    X_train, X_test, Y_train, Y_test, meta_train, meta_test = train_test_split(
        X, Y, metadata, test_size=0.2, random_state=42, stratify=strat_labels
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train and evaluate models for each target metric
    print("\n[4] Training and evaluating models...")
    all_results = []
    importance_dfs = {}
    
    for i, metric in enumerate(TARGET_METRICS):
        print(f"\n    Target: {metric}")
        
        y_train = Y_train[:, i]
        y_test = Y_test[:, i]
        
        results_df, best_model, best_model_name, scaler = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, metric
        )
        all_results.append(results_df)
        
        best_r2 = results_df['r2_score'].max()
        print(f"      Best: {best_model_name} (R² = {best_r2:.4f})")
        
        # Extract feature importance
        importance = extract_feature_importance(best_model, best_model_name, X.shape[1])
        importance_dfs[metric] = importance
    
    # Combine results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(output_dir / 'rq11_model_performance.csv', index=False)
    print(f"\n    Exported: rq11_model_performance.csv")
    
    # Export feature importance
    print("\n[5] Exporting feature importance...")
    for metric, imp_df in importance_dfs.items():
        if imp_df is not None:
            safe_name = metric.lower().replace(' ', '_')
            imp_df.to_csv(output_dir / f'rq11_feature_importance_{safe_name}.csv', index=False)
    print(f"    Exported {len([v for v in importance_dfs.values() if v is not None])} importance files")
    
    # Cross-model analysis
    print("\n[6] Performing cross-model/culture analysis...")
    cross_analysis = perform_cross_model_analysis(X, Y, metadata)
    cross_analysis.to_csv(output_dir / 'rq11_cross_analysis.csv', index=False)
    print(f"    Exported: rq11_cross_analysis.csv")
    
    # Correlation analysis
    print("\n[7] Computing embedding-score correlations...")
    correlations = compute_embedding_score_correlation(X, Y)
    correlations.to_csv(output_dir / 'rq11_dimension_correlations.csv', index=False)
    print(f"    Exported: rq11_dimension_correlations.csv")
    
    n_sig = correlations['significant'].sum()
    print(f"    Significant correlations: {n_sig}/{len(correlations)} ({n_sig/len(correlations)*100:.1f}%)")
    
    # Create visualizations
    print("\n[8] Creating visualizations...")
    create_visualizations(all_results_df, cross_analysis, correlations, importance_dfs, output_dir)
    
    # Generate summary
    print("\n[9] Generating summary...")
    summary = generate_summary(all_results_df, cross_analysis, correlations)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq11_summary.csv', index=False)
    print(f"    Exported: rq11_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ11 ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nKey Findings:")
    
    print(f"\n  Model Performance (Best R² per Target):")
    for metric in TARGET_METRICS:
        metric_best = all_results_df[all_results_df['target_metric'] == metric].nlargest(1, 'r2_score').iloc[0]
        emoji = "🟢" if metric_best['r2_score'] > 0.1 else "🟡" if metric_best['r2_score'] > 0 else "🔴"
        short_name = METRIC_SHORT_NAMES.get(metric, metric)
        print(f"    {emoji} {short_name}: R² = {metric_best['r2_score']:.4f} ({metric_best['model']})")
    
    best_overall = all_results_df.nlargest(1, 'r2_score').iloc[0]
    worst_overall = all_results_df.nsmallest(1, 'r2_score').iloc[0]
    
    best_short = METRIC_SHORT_NAMES.get(best_overall['target_metric'], best_overall['target_metric'])
    worst_short = METRIC_SHORT_NAMES.get(worst_overall['target_metric'], worst_overall['target_metric'])
    
    print(f"\n  Best Prediction: {best_short} with {best_overall['model']}")
    print(f"    R² = {best_overall['r2_score']:.4f}, MAE = {best_overall['mae']:.4f}")
    
    print(f"\n  Worst Prediction: {worst_short} with {worst_overall['model']}")
    print(f"    R² = {worst_overall['r2_score']:.4f}")
    
    avg_r2 = all_results_df['r2_score'].mean()
    if avg_r2 > 0.1:
        print(f"\n  ✓ Embeddings have MODERATE predictive power (avg R² = {avg_r2:.4f})")
        print(f"    Story embeddings capture some quality-related features.")
    elif avg_r2 > 0:
        print(f"\n  ⚠️  Embeddings have WEAK predictive power (avg R² = {avg_r2:.4f})")
        print(f"    Limited ability to predict human scores from embeddings alone.")
    else:
        print(f"\n  ✗ Embeddings have NO predictive power (avg R² = {avg_r2:.4f})")
        print(f"    Human judgments based on factors not captured in embeddings.")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq11_model_performance.csv - ML model results")
    print("  2. rq11_cross_analysis.csv - By model/culture analysis")
    print("  3. rq11_dimension_correlations.csv - Dimension-score correlations")
    print("  4. rq11_summary.csv - Key findings")
    print("  5-8. rq11_feature_importance_*.csv - Top important dimensions")
    print("-" * 70)
    
    return all_results_df, cross_analysis, correlations


if __name__ == "__main__":
    all_results, cross_analysis, correlations = main()
