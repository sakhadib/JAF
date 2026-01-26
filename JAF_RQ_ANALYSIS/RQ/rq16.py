"""
RQ16: Regional-Cultural Consistency
====================================
Question: Are cultures in the same region appropriately distinct?
          Do LLMs capture meaningful differences between cultures that share a geographic region?

Method:
- Filter by region (e.g., "Chittagong Hill Tracts" with Chakma, Marma, Tripura)
- Compute distribution overlap between culture embeddings within each region
- Statistical test: KL Divergence, MMD, Wasserstein distance between distributions
- Repeat for all multi-culture regions

Data Source: Parquet (story_embedding) + CSV (evaluation scores)

Regions with multiple cultures:
- Chittagong Hill Tracts: Chakma, Marma, Tripura
- Sylhet (border areas): Khasi, Manipuri (Meitei)

Interpretation:
- High distinctiveness = LLM captures cultural nuances even within same region
- Low distinctiveness = LLM may be conflating cultures in same region
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, cosine
from scipy.stats import entropy, ks_2samp
from pathlib import Path
from itertools import combinations
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

# Try to import sklearn for dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available.")


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


def identify_multi_culture_regions(df):
    """Identify regions with multiple cultures."""
    region_cultures = df.groupby('region')['culture'].nunique()
    multi_regions = region_cultures[region_cultures > 1].index.tolist()
    
    region_info = []
    for region in multi_regions:
        cultures = df[df['region'] == region]['culture'].unique().tolist()
        n_stories = len(df[df['region'] == region])
        region_info.append({
            'region': region,
            'cultures': cultures,
            'n_cultures': len(cultures),
            'n_stories': n_stories
        })
    
    return pd.DataFrame(region_info)


def compute_culture_centroids(df, region):
    """Compute centroid for each culture within a region."""
    region_df = df[df['region'] == region]
    centroids = {}
    
    for culture in region_df['culture'].unique():
        culture_embeddings = np.array(region_df[region_df['culture'] == culture]['story_embedding'].tolist())
        centroids[culture] = culture_embeddings.mean(axis=0)
    
    return centroids


def compute_pairwise_culture_distances(df, region):
    """Compute pairwise distances between cultures within a region."""
    region_df = df[df['region'] == region]
    cultures = region_df['culture'].unique()
    
    results = []
    
    for c1, c2 in combinations(cultures, 2):
        emb1 = np.array(region_df[region_df['culture'] == c1]['story_embedding'].tolist())
        emb2 = np.array(region_df[region_df['culture'] == c2]['story_embedding'].tolist())
        
        # Centroid distance
        centroid1 = emb1.mean(axis=0)
        centroid2 = emb2.mean(axis=0)
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        cosine_dist = cosine(centroid1, centroid2)
        
        # Mean pairwise distance
        pairwise_distances = cdist(emb1, emb2, metric='cosine')
        mean_pairwise = pairwise_distances.mean()
        min_pairwise = pairwise_distances.min()
        max_pairwise = pairwise_distances.max()
        
        # Distribution overlap via KS test
        # Project to 1D using PCA for KS test
        combined = np.vstack([emb1, emb2])
        if HAS_SKLEARN and len(combined) > 2:
            pca = PCA(n_components=1)
            projected = pca.fit_transform(combined)
            proj1 = projected[:len(emb1)].flatten()
            proj2 = projected[len(emb1):].flatten()
            ks_stat, ks_pvalue = ks_2samp(proj1, proj2)
        else:
            ks_stat, ks_pvalue = np.nan, np.nan
        
        # MMD (Maximum Mean Discrepancy) - simplified Gaussian kernel
        mmd = compute_mmd(emb1, emb2)
        
        results.append({
            'region': region,
            'culture_1': c1,
            'culture_2': c2,
            'n_stories_1': len(emb1),
            'n_stories_2': len(emb2),
            'centroid_euclidean_distance': centroid_distance,
            'centroid_cosine_distance': cosine_dist,
            'mean_pairwise_cosine': mean_pairwise,
            'min_pairwise_cosine': min_pairwise,
            'max_pairwise_cosine': max_pairwise,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'distributions_distinct': ks_pvalue < 0.05 if not np.isnan(ks_pvalue) else None,
            'mmd_score': mmd
        })
    
    return pd.DataFrame(results)


def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between two distributions.
    Uses RBF kernel.
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    
    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)
    
    # RBF kernel
    gamma = 1.0 / X.shape[1]  # Use 1/dim as gamma
    
    K_XX = np.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_YY = np.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    K_XY = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))
    
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd


def compute_intra_culture_consistency(df, region):
    """Compute how consistent stories are within each culture."""
    region_df = df[df['region'] == region]
    results = []
    
    for culture in region_df['culture'].unique():
        culture_df = region_df[region_df['culture'] == culture]
        embeddings = np.array(culture_df['story_embedding'].tolist())
        
        if len(embeddings) < 2:
            continue
        
        # Compute pairwise similarities within culture
        if HAS_SKLEARN:
            sim_matrix = cosine_similarity(embeddings)
            # Get upper triangle (excluding diagonal)
            upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
            mean_intra_sim = upper_tri.mean()
            std_intra_sim = upper_tri.std()
        else:
            mean_intra_sim = np.nan
            std_intra_sim = np.nan
        
        # Compute distance to centroid
        centroid = embeddings.mean(axis=0)
        distances_to_centroid = [cosine(emb, centroid) for emb in embeddings]
        
        # Quality scores
        mean_cultural = culture_df['Cultural Accuracy & Authenticity'].mean()
        
        results.append({
            'region': region,
            'culture': culture,
            'n_stories': len(embeddings),
            'mean_intra_similarity': mean_intra_sim,
            'std_intra_similarity': std_intra_sim,
            'mean_centroid_distance': np.mean(distances_to_centroid),
            'std_centroid_distance': np.std(distances_to_centroid),
            'mean_cultural_accuracy': mean_cultural
        })
    
    return pd.DataFrame(results)


def compute_regional_distinctiveness_score(pairwise_df):
    """
    Compute an overall distinctiveness score for a region.
    Higher = cultures are more distinct.
    """
    if len(pairwise_df) == 0:
        return None
    
    results = {
        'region': pairwise_df['region'].iloc[0],
        'n_culture_pairs': len(pairwise_df),
        'mean_centroid_distance': pairwise_df['centroid_cosine_distance'].mean(),
        'mean_pairwise_distance': pairwise_df['mean_pairwise_cosine'].mean(),
        'pct_statistically_distinct': pairwise_df['distributions_distinct'].mean() * 100 if pairwise_df['distributions_distinct'].notna().any() else np.nan,
        'mean_mmd': pairwise_df['mmd_score'].mean(),
        'min_mmd': pairwise_df['mmd_score'].min(),
        'max_mmd': pairwise_df['mmd_score'].max()
    }
    
    # Distinctiveness score (0-1, higher = more distinct)
    # Combine multiple metrics
    distinctiveness = (
        results['mean_centroid_distance'] * 0.3 +
        results['mean_pairwise_distance'] * 0.3 +
        (results['pct_statistically_distinct'] / 100 if not np.isnan(results['pct_statistically_distinct']) else 0.5) * 0.4
    )
    results['distinctiveness_score'] = distinctiveness
    
    return results


def analyze_by_model(df, region):
    """Analyze distinctiveness by generation model."""
    region_df = df[df['region'] == region]
    results = []
    
    for model in region_df['model'].unique():
        model_df = region_df[region_df['model'] == model]
        cultures = model_df['culture'].unique()
        
        if len(cultures) < 2:
            continue
        
        # Compute culture centroids
        centroids = {}
        for culture in cultures:
            emb = np.array(model_df[model_df['culture'] == culture]['story_embedding'].tolist())
            if len(emb) > 0:
                centroids[culture] = emb.mean(axis=0)
        
        # Pairwise distances
        distances = []
        for c1, c2 in combinations(centroids.keys(), 2):
            dist = cosine(centroids[c1], centroids[c2])
            distances.append(dist)
        
        if distances:
            results.append({
                'region': region,
                'model': model,
                'n_cultures': len(cultures),
                'n_stories': len(model_df),
                'mean_inter_culture_distance': np.mean(distances),
                'std_inter_culture_distance': np.std(distances) if len(distances) > 1 else 0
            })
    
    return pd.DataFrame(results)


def compare_intra_vs_inter_region(df):
    """
    Compare distances within regions vs across regions.
    If intra-region distance < inter-region distance, cultures share regional characteristics.
    """
    # Get unique regions and their cultures
    region_cultures = df.groupby('region')['culture'].unique().to_dict()
    multi_culture_regions = {r: c for r, c in region_cultures.items() if len(c) > 1}
    
    # Compute culture centroids
    culture_centroids = {}
    for culture in df['culture'].unique():
        emb = np.array(df[df['culture'] == culture]['story_embedding'].tolist())
        culture_centroids[culture] = emb.mean(axis=0)
    
    results = []
    
    for region, cultures in multi_culture_regions.items():
        # Intra-region distances (between cultures in same region)
        intra_distances = []
        for c1, c2 in combinations(cultures, 2):
            if c1 in culture_centroids and c2 in culture_centroids:
                dist = cosine(culture_centroids[c1], culture_centroids[c2])
                intra_distances.append(dist)
        
        # Inter-region distances (between cultures in this region vs other regions)
        inter_distances = []
        other_cultures = [c for c in df['culture'].unique() if c not in cultures]
        for c1 in cultures:
            for c2 in other_cultures:
                if c1 in culture_centroids and c2 in culture_centroids:
                    dist = cosine(culture_centroids[c1], culture_centroids[c2])
                    inter_distances.append(dist)
        
        if intra_distances and inter_distances:
            # Statistical test
            t_stat, p_value = stats.ttest_ind(intra_distances, inter_distances)
            
            results.append({
                'region': region,
                'n_cultures_in_region': len(cultures),
                'mean_intra_region_distance': np.mean(intra_distances),
                'std_intra_region_distance': np.std(intra_distances) if len(intra_distances) > 1 else 0,
                'mean_inter_region_distance': np.mean(inter_distances),
                'std_inter_region_distance': np.std(inter_distances),
                'intra_vs_inter_ratio': np.mean(intra_distances) / np.mean(inter_distances),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05,
                'regional_clustering': np.mean(intra_distances) < np.mean(inter_distances)
            })
    
    return pd.DataFrame(results)


def analyze_quality_by_distinctiveness(df, pairwise_results):
    """Correlate cultural distinctiveness with quality scores."""
    results = []
    
    for region in pairwise_results['region'].unique():
        region_df = df[df['region'] == region]
        region_pairwise = pairwise_results[pairwise_results['region'] == region]
        
        # Get distinctiveness metric
        mean_dist = region_pairwise['centroid_cosine_distance'].mean()
        
        # Get quality scores
        for culture in region_df['culture'].unique():
            culture_df = region_df[region_df['culture'] == culture]
            
            results.append({
                'region': region,
                'culture': culture,
                'n_stories': len(culture_df),
                'region_distinctiveness': mean_dist,
                'cultural_accuracy': culture_df['Cultural Accuracy & Authenticity'].mean(),
                'contextual_appropriateness': culture_df['Contextual & Temporal Appropriateness'].mean(),
                'narrative_coherence': culture_df['Narrative & Symbolic Coherence'].mean(),
                'linguistic_fluency': culture_df['Linguistic & Expressive Appropriateness (Bangla)'].mean(),
                'average_score': culture_df[TARGET_METRICS].mean(axis=1).mean()
            })
    
    return pd.DataFrame(results)


def create_visualizations(df, multi_regions, pairwise_all, intra_culture, 
                         model_analysis, intra_inter, output_dir):
    """Create all visualizations."""
    if not HAS_PLOTTING:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # 1. Heatmap of pairwise distances per region
    for region in multi_regions['region']:
        region_pairs = pairwise_all[pairwise_all['region'] == region]
        if len(region_pairs) == 0:
            continue
        
        # Create distance matrix
        cultures = list(set(region_pairs['culture_1'].tolist() + region_pairs['culture_2'].tolist()))
        n = len(cultures)
        dist_matrix = np.zeros((n, n))
        
        for _, row in region_pairs.iterrows():
            i = cultures.index(row['culture_1'])
            j = cultures.index(row['culture_2'])
            dist_matrix[i, j] = row['centroid_cosine_distance']
            dist_matrix[j, i] = row['centroid_cosine_distance']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dist_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=cultures, yticklabels=cultures, ax=ax)
        ax.set_title(f'RQ16: Cultural Distance Matrix\n{region}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        safe_region = region.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
        plt.savefig(output_dir / f'rq16_distance_matrix_{safe_region}.png', bbox_inches='tight')
        plt.close()
    
    # 2. Regional distinctiveness comparison
    if len(pairwise_all) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        region_stats = pairwise_all.groupby('region')['centroid_cosine_distance'].agg(['mean', 'std']).reset_index()
        region_stats = region_stats.sort_values('mean', ascending=False)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(region_stats)))
        bars = ax.bar(range(len(region_stats)), region_stats['mean'], 
                     yerr=region_stats['std'], capsize=5, color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(region_stats)))
        ax.set_xticklabels(region_stats['region'], rotation=45, ha='right')
        ax.set_ylabel('Mean Inter-Culture Distance', fontsize=11)
        ax.set_title('RQ16: Cultural Distinctiveness by Region\n(Higher = More Distinct Cultures)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq16_regional_distinctiveness.png', bbox_inches='tight')
        plt.close()
    
    # 3. Intra vs Inter region comparison
    if len(intra_inter) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(intra_inter))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, intra_inter['mean_intra_region_distance'], width, 
                      label='Intra-region', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, intra_inter['mean_inter_region_distance'], width,
                      label='Inter-region', color='coral', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(intra_inter['region'], rotation=45, ha='right')
        ax.set_ylabel('Mean Cosine Distance', fontsize=11)
        ax.set_title('RQ16: Intra-Region vs Inter-Region Cultural Distances', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars2, intra_inter['significant_difference'])):
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '*',
                       ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq16_intra_vs_inter_region.png', bbox_inches='tight')
        plt.close()
    
    # 4. Model comparison for distinctiveness
    if len(model_analysis) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_agg = model_analysis.groupby('model')['mean_inter_culture_distance'].mean().sort_values(ascending=False)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_agg)))
        bars = ax.bar(range(len(model_agg)), model_agg.values, color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(model_agg)))
        ax.set_xticklabels(model_agg.index, rotation=45, ha='right')
        ax.set_ylabel('Mean Inter-Culture Distance', fontsize=11)
        ax.set_title('RQ16: Cultural Distinctiveness by Generation Model', 
                    fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, model_agg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq16_distinctiveness_by_model.png', bbox_inches='tight')
        plt.close()
    
    # 5. Intra-culture consistency
    if len(intra_culture) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_data = intra_culture.sort_values('mean_intra_similarity', ascending=False)
        colors = [plt.cm.tab10(i % 10) for i in range(len(sorted_data))]
        
        bars = ax.bar(range(len(sorted_data)), sorted_data['mean_intra_similarity'], 
                     color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(sorted_data)))
        labels = [f"{row['culture']}\n({row['region'][:15]}...)" if len(row['region']) > 15 
                 else f"{row['culture']}\n({row['region']})" 
                 for _, row in sorted_data.iterrows()]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Intra-Culture Similarity', fontsize=11)
        ax.set_title('RQ16: Within-Culture Story Consistency\n(Higher = More Consistent)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq16_intra_culture_consistency.png', bbox_inches='tight')
        plt.close()
    
    # 6. Scatter: Distinctiveness vs Quality
    if len(pairwise_all) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aggregate by region
        region_quality = df.groupby('region')[TARGET_METRICS].mean().mean(axis=1)
        region_dist = pairwise_all.groupby('region')['centroid_cosine_distance'].mean()
        
        common_regions = region_quality.index.intersection(region_dist.index)
        
        ax.scatter(region_dist[common_regions], region_quality[common_regions], 
                  s=150, c='steelblue', edgecolors='black')
        
        for region in common_regions:
            ax.annotate(region[:20], (region_dist[region], region_quality[region]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        ax.set_xlabel('Mean Cultural Distinctiveness', fontsize=11)
        ax.set_ylabel('Mean Quality Score', fontsize=11)
        ax.set_title('RQ16: Regional Distinctiveness vs. Quality Scores', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq16_distinctiveness_vs_quality.png', bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(multi_regions, pairwise_all, intra_inter, distinctiveness_scores):
    """Generate summary of key findings."""
    summary = {
        'research_question': 'RQ16: Regional-Cultural Consistency',
        'main_question': 'Are cultures in the same region appropriately distinct?',
        'n_multi_culture_regions': len(multi_regions),
        'total_culture_pairs_analyzed': len(pairwise_all)
    }
    
    if len(pairwise_all) > 0:
        summary['overall_mean_inter_culture_distance'] = pairwise_all['centroid_cosine_distance'].mean()
        summary['overall_pct_statistically_distinct'] = pairwise_all['distributions_distinct'].mean() * 100 if pairwise_all['distributions_distinct'].notna().any() else np.nan
        
        # Most distinct region
        region_means = pairwise_all.groupby('region')['centroid_cosine_distance'].mean()
        summary['most_distinct_region'] = region_means.idxmax()
        summary['most_distinct_distance'] = region_means.max()
        
        # Least distinct region  
        summary['least_distinct_region'] = region_means.idxmin()
        summary['least_distinct_distance'] = region_means.min()
    
    if len(intra_inter) > 0:
        # Regional clustering effect
        summary['pct_regions_with_clustering'] = intra_inter['regional_clustering'].mean() * 100
        summary['mean_intra_inter_ratio'] = intra_inter['intra_vs_inter_ratio'].mean()
    
    return summary


def main():
    """Main execution function for RQ16 analysis."""
    print("=" * 70)
    print("RQ16: Regional-Cultural Consistency")
    print("=" * 70)
    print("\nQuestion: Are cultures in the same region appropriately distinct?")
    print("Method: Analyze embedding distributions within multi-culture regions")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading and merging data...")
    df = load_data()
    print(f"    Loaded {len(df)} stories")
    
    # Identify multi-culture regions
    print("\n[2] Identifying multi-culture regions...")
    multi_regions = identify_multi_culture_regions(df)
    multi_regions.to_csv(output_dir / 'rq16_multi_culture_regions.csv', index=False)
    print(f"    Exported: rq16_multi_culture_regions.csv")
    
    print(f"\n    Found {len(multi_regions)} regions with multiple cultures:")
    for _, row in multi_regions.iterrows():
        print(f"      • {row['region']}: {', '.join(row['cultures'])} ({row['n_stories']} stories)")
    
    # Compute pairwise distances for each region
    print("\n[3] Computing pairwise cultural distances...")
    pairwise_all = []
    distinctiveness_scores = []
    
    for _, row in multi_regions.iterrows():
        region = row['region']
        print(f"    Processing: {region}")
        
        pairwise = compute_pairwise_culture_distances(df, region)
        if len(pairwise) > 0:
            pairwise_all.append(pairwise)
            
            # Distinctiveness score
            dist_score = compute_regional_distinctiveness_score(pairwise)
            if dist_score:
                distinctiveness_scores.append(dist_score)
    
    if pairwise_all:
        pairwise_all = pd.concat(pairwise_all, ignore_index=True)
        pairwise_all.to_csv(output_dir / 'rq16_pairwise_culture_distances.csv', index=False)
        print(f"\n    Exported: rq16_pairwise_culture_distances.csv ({len(pairwise_all)} pairs)")
        
        print("\n    Pairwise Culture Distances:")
        for _, row in pairwise_all.iterrows():
            distinct = "✓" if row['distributions_distinct'] else "✗" if row['distributions_distinct'] is not None else "?"
            print(f"      {row['culture_1']} ↔ {row['culture_2']}: "
                  f"cosine={row['centroid_cosine_distance']:.4f}, "
                  f"distinct={distinct}")
    else:
        pairwise_all = pd.DataFrame()
        print("    No pairwise comparisons could be made")
    
    # Distinctiveness scores
    if distinctiveness_scores:
        dist_df = pd.DataFrame(distinctiveness_scores)
        dist_df.to_csv(output_dir / 'rq16_distinctiveness_scores.csv', index=False)
        print(f"\n    Exported: rq16_distinctiveness_scores.csv")
    
    # Intra-culture consistency
    print("\n[4] Computing intra-culture consistency...")
    intra_culture_all = []
    
    for _, row in multi_regions.iterrows():
        intra = compute_intra_culture_consistency(df, row['region'])
        if len(intra) > 0:
            intra_culture_all.append(intra)
    
    if intra_culture_all:
        intra_culture = pd.concat(intra_culture_all, ignore_index=True)
        intra_culture.to_csv(output_dir / 'rq16_intra_culture_consistency.csv', index=False)
        print(f"    Exported: rq16_intra_culture_consistency.csv")
    else:
        intra_culture = pd.DataFrame()
    
    # Model analysis
    print("\n[5] Analyzing distinctiveness by model...")
    model_analysis_all = []
    
    for _, row in multi_regions.iterrows():
        model_df = analyze_by_model(df, row['region'])
        if len(model_df) > 0:
            model_analysis_all.append(model_df)
    
    if model_analysis_all:
        model_analysis = pd.concat(model_analysis_all, ignore_index=True)
        model_analysis.to_csv(output_dir / 'rq16_model_distinctiveness.csv', index=False)
        print(f"    Exported: rq16_model_distinctiveness.csv")
        
        print("\n    Model Distinctiveness Ranking:")
        model_rank = model_analysis.groupby('model')['mean_inter_culture_distance'].mean().sort_values(ascending=False)
        for model, dist in model_rank.items():
            print(f"      {model}: {dist:.4f}")
    else:
        model_analysis = pd.DataFrame()
    
    # Intra vs Inter region comparison
    print("\n[6] Comparing intra vs inter-region distances...")
    intra_inter = compare_intra_vs_inter_region(df)
    intra_inter.to_csv(output_dir / 'rq16_intra_vs_inter_region.csv', index=False)
    print(f"    Exported: rq16_intra_vs_inter_region.csv")
    
    if len(intra_inter) > 0:
        print("\n    Regional Clustering Analysis:")
        for _, row in intra_inter.iterrows():
            cluster = "✓ regional clustering" if row['regional_clustering'] else "✗ no clustering"
            sig = "*" if row['significant_difference'] else ""
            print(f"      {row['region']}: intra={row['mean_intra_region_distance']:.4f}, "
                  f"inter={row['mean_inter_region_distance']:.4f} → {cluster} {sig}")
    
    # Quality analysis
    print("\n[7] Analyzing distinctiveness vs quality...")
    if len(pairwise_all) > 0:
        quality_analysis = analyze_quality_by_distinctiveness(df, pairwise_all)
        quality_analysis.to_csv(output_dir / 'rq16_distinctiveness_quality.csv', index=False)
        print(f"    Exported: rq16_distinctiveness_quality.csv")
    
    # Create visualizations
    print("\n[8] Creating visualizations...")
    create_visualizations(df, multi_regions, pairwise_all, intra_culture,
                         model_analysis, intra_inter, output_dir)
    
    # Generate summary
    print("\n[9] Generating summary...")
    summary = generate_summary(multi_regions, pairwise_all, intra_inter, 
                              distinctiveness_scores if distinctiveness_scores else [])
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq16_summary.csv', index=False)
    print(f"    Exported: rq16_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ16 ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nKey Findings:")
    
    print(f"\n  Multi-Culture Regions Analyzed: {len(multi_regions)}")
    for _, row in multi_regions.iterrows():
        print(f"    • {row['region']}: {row['n_cultures']} cultures")
    
    if len(pairwise_all) > 0:
        print(f"\n  Cultural Distinctiveness Within Regions:")
        region_dist = pairwise_all.groupby('region')['centroid_cosine_distance'].mean().sort_values(ascending=False)
        for region, dist in region_dist.items():
            print(f"    • {region}: {dist:.4f}")
        
        pct_distinct = pairwise_all['distributions_distinct'].mean() * 100 if pairwise_all['distributions_distinct'].notna().any() else 0
        print(f"\n  Statistical Distinctiveness:")
        print(f"    {pct_distinct:.1f}% of culture pairs are statistically distinct (KS test p<0.05)")
    
    if len(intra_inter) > 0:
        pct_cluster = intra_inter['regional_clustering'].mean() * 100
        print(f"\n  Regional Clustering Effect:")
        print(f"    {pct_cluster:.1f}% of regions show clustering (intra < inter distance)")
        
        if pct_cluster < 50:
            print(f"    → Cultures within regions are NOT more similar than across regions")
            print(f"    → LLMs capture cultural distinctions regardless of geographic proximity")
        else:
            print(f"    → Cultures within regions ARE more similar")
            print(f"    → LLMs may conflate geographically close cultures")
    
    if len(model_analysis) > 0:
        best_model = model_analysis.groupby('model')['mean_inter_culture_distance'].mean().idxmax()
        worst_model = model_analysis.groupby('model')['mean_inter_culture_distance'].mean().idxmin()
        print(f"\n  Model Comparison:")
        print(f"    Most distinctive: {best_model}")
        print(f"    Least distinctive: {worst_model}")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq16_multi_culture_regions.csv - Regions with multiple cultures")
    print("  2. rq16_pairwise_culture_distances.csv - Pairwise cultural distances")
    print("  3. rq16_distinctiveness_scores.csv - Regional distinctiveness scores")
    print("  4. rq16_intra_culture_consistency.csv - Within-culture consistency")
    print("  5. rq16_model_distinctiveness.csv - Distinctiveness by model")
    print("  6. rq16_intra_vs_inter_region.csv - Intra vs inter-region comparison")
    print("  7. rq16_distinctiveness_quality.csv - Quality vs distinctiveness")
    print("  8. rq16_summary.csv - Key findings")
    print("-" * 70)
    
    return df, pairwise_all, intra_inter


if __name__ == "__main__":
    df, pairwise_all, intra_inter = main()
