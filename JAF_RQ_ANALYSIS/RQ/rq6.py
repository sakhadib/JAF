"""
RQ6: Semantic Separability of Cultural Narratives
==================================================
Question: Do story embeddings cluster by culture or collapse into generic groups?

Method:
- Load story_embedding from parquet (openai-large as primary)
- Dimensionality reduction: UMAP and t-SNE to 2D
- Cluster quality: Silhouette Score by culture labels
- Visualize: Scatter plot colored by culture
- Pairwise analysis: Which cultures overlap most?

Data Source: Parquet only (story_embedding vectors)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import dimensionality reduction libraries
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("Warning: sklearn TSNE not available.")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")


def load_parquet_data(embedding_model='openai_text-embedding-3-large'):
    """Load parquet data with story embeddings."""
    parquet_dir = Path(__file__).parent.parent / "data" / "parquet"
    parquet_file = parquet_dir / f"{embedding_model}.parquet"
    
    if not parquet_file.exists():
        # Try to find any parquet file
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/parquet/")
        parquet_file = parquet_files[0]
        print(f"    Using fallback: {parquet_file.name}")
    
    df = pd.read_parquet(parquet_file)
    return df


def extract_embeddings(df):
    """Extract story embeddings as a numpy array."""
    # Convert list/array column to numpy matrix
    embeddings = np.array(df['story_embedding'].tolist())
    return embeddings


def perform_tsne(embeddings, perplexity=30, max_iter=1000, random_state=42):
    """Perform t-SNE dimensionality reduction to 2D."""
    if not HAS_TSNE:
        return None
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


def perform_umap(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Perform UMAP dimensionality reduction to 2D."""
    if not HAS_UMAP:
        return None
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine'
    )
    
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def calculate_silhouette_scores(embeddings, labels):
    """Calculate silhouette score for clustering quality."""
    # Overall silhouette score
    overall_score = silhouette_score(embeddings, labels, metric='cosine')
    
    # Per-sample silhouette scores
    sample_scores = silhouette_samples(embeddings, labels, metric='cosine')
    
    return overall_score, sample_scores


def calculate_per_culture_silhouette(embeddings, labels):
    """Calculate average silhouette score per culture."""
    sample_scores = silhouette_samples(embeddings, labels, metric='cosine')
    
    results = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        culture_scores = sample_scores[mask]
        results.append({
            'culture': label,
            'n': np.sum(mask),
            'silhouette_mean': np.mean(culture_scores),
            'silhouette_std': np.std(culture_scores),
            'silhouette_min': np.min(culture_scores),
            'silhouette_max': np.max(culture_scores)
        })
    
    return pd.DataFrame(results).sort_values('silhouette_mean', ascending=False)


def calculate_pairwise_culture_distances(embeddings, labels):
    """Calculate average pairwise distances between cultures."""
    unique_cultures = np.unique(labels)
    n_cultures = len(unique_cultures)
    
    # Calculate centroid for each culture
    centroids = {}
    for culture in unique_cultures:
        mask = labels == culture
        centroids[culture] = embeddings[mask].mean(axis=0)
    
    # Pairwise distances between centroids
    results = []
    for i, c1 in enumerate(unique_cultures):
        for j, c2 in enumerate(unique_cultures):
            if i < j:
                # Cosine distance
                cos_dist = 1 - np.dot(centroids[c1], centroids[c2]) / (
                    np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]))
                
                # Euclidean distance
                euc_dist = np.linalg.norm(centroids[c1] - centroids[c2])
                
                results.append({
                    'culture_1': c1,
                    'culture_2': c2,
                    'cosine_distance': cos_dist,
                    'euclidean_distance': euc_dist
                })
    
    return pd.DataFrame(results).sort_values('cosine_distance')


def calculate_intra_culture_variance(embeddings, labels):
    """Calculate variance within each culture cluster."""
    results = []
    
    for culture in np.unique(labels):
        mask = labels == culture
        culture_embeddings = embeddings[mask]
        
        # Calculate centroid
        centroid = culture_embeddings.mean(axis=0)
        
        # Distances from centroid
        distances = np.linalg.norm(culture_embeddings - centroid, axis=1)
        
        # Variance metrics
        results.append({
            'culture': culture,
            'n': np.sum(mask),
            'intra_cluster_variance': np.var(distances),
            'intra_cluster_std': np.std(distances),
            'mean_dist_to_centroid': np.mean(distances),
            'max_dist_to_centroid': np.max(distances)
        })
    
    return pd.DataFrame(results).sort_values('intra_cluster_variance')


def identify_overlapping_cultures(pairwise_distances, threshold_percentile=25):
    """Identify which cultures have the most semantic overlap."""
    # Cultures with smallest distances are most similar/overlapping
    threshold = np.percentile(pairwise_distances['cosine_distance'], threshold_percentile)
    overlapping = pairwise_distances[pairwise_distances['cosine_distance'] <= threshold]
    
    return overlapping


def identify_distinct_cultures(pairwise_distances, threshold_percentile=75):
    """Identify which cultures are most semantically distinct."""
    threshold = np.percentile(pairwise_distances['cosine_distance'], threshold_percentile)
    distinct = pairwise_distances[pairwise_distances['cosine_distance'] >= threshold]
    
    return distinct


def create_distance_matrix(pairwise_distances, cultures):
    """Create a symmetric distance matrix from pairwise distances."""
    n = len(cultures)
    matrix = np.zeros((n, n))
    
    culture_to_idx = {c: i for i, c in enumerate(cultures)}
    
    for _, row in pairwise_distances.iterrows():
        i = culture_to_idx[row['culture_1']]
        j = culture_to_idx[row['culture_2']]
        matrix[i, j] = row['cosine_distance']
        matrix[j, i] = row['cosine_distance']
    
    return pd.DataFrame(matrix, index=cultures, columns=cultures)


def create_visualizations(df, embeddings, tsne_2d, umap_2d, per_culture_sil, distance_matrix, output_dir):
    """Create and save all visualizations."""
    if not HAS_PLOTTING:
        return
    
    cultures = df['culture'].values
    unique_cultures = df['culture'].unique()
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # Color palette for cultures
    palette = sns.color_palette("husl", len(unique_cultures))
    color_map = {c: palette[i] for i, c in enumerate(unique_cultures)}
    
    # 1. t-SNE scatter plot
    if tsne_2d is not None:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for culture in unique_cultures:
            mask = cultures == culture
            ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], 
                      c=[color_map[culture]], label=culture, alpha=0.6, s=50)
        
        ax.set_title('RQ6: Story Embeddings - t-SNE Visualization by Culture', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.legend(title='Culture', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq6_tsne_by_culture.png', bbox_inches='tight')
        plt.close()
    
    # 2. UMAP scatter plot
    if umap_2d is not None:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for culture in unique_cultures:
            mask = cultures == culture
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], 
                      c=[color_map[culture]], label=culture, alpha=0.6, s=50)
        
        ax.set_title('RQ6: Story Embeddings - UMAP Visualization by Culture', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        ax.legend(title='Culture', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq6_umap_by_culture.png', bbox_inches='tight')
        plt.close()
    
    # 3. Silhouette score bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    per_culture_sorted = per_culture_sil.sort_values('silhouette_mean', ascending=True)
    
    colors = ['#e74c3c' if s < 0 else '#3498db' for s in per_culture_sorted['silhouette_mean']]
    
    bars = ax.barh(range(len(per_culture_sorted)), per_culture_sorted['silhouette_mean'], color=colors)
    ax.set_yticks(range(len(per_culture_sorted)))
    ax.set_yticklabels(per_culture_sorted['culture'])
    ax.set_xlabel('Silhouette Score', fontsize=11)
    ax.set_ylabel('Culture', fontsize=11)
    ax.set_title('RQ6: Silhouette Score by Culture\n(Higher = Better Separation)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=per_culture_sorted['silhouette_mean'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f"Mean: {per_culture_sorted['silhouette_mean'].mean():.3f}")
    ax.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, per_culture_sorted['silhouette_mean'])):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq6_silhouette_by_culture.png', bbox_inches='tight')
    plt.close()
    
    # 4. Distance heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        ax=ax,
        cbar_kws={'label': 'Cosine Distance'}
    )
    ax.set_title('RQ6: Pairwise Cultural Distance Matrix\n(Lower = More Similar)', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rq6_distance_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 5. Scatter by model (using t-SNE or UMAP)
    embed_2d = umap_2d if umap_2d is not None else tsne_2d
    method_name = 'UMAP' if umap_2d is not None else 't-SNE'
    
    if embed_2d is not None:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        models = df['model'].values
        unique_models = df['model'].unique()
        model_palette = sns.color_palette("Set2", len(unique_models))
        model_color_map = {m: model_palette[i] for i, m in enumerate(unique_models)}
        
        for model in unique_models:
            mask = models == model
            ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1], 
                      c=[model_color_map[model]], label=model, alpha=0.6, s=50)
        
        ax.set_title(f'RQ6: Story Embeddings - {method_name} by Model', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rq6_embedding_by_model.png', bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to result/ folder")


def generate_summary(df, overall_silhouette, per_culture_sil, pairwise_distances, intra_variance):
    """Generate summary of key findings."""
    # Most separable and most overlapping cultures
    best_separated = per_culture_sil.iloc[0]
    most_overlapping = per_culture_sil.iloc[-1]
    
    # Most similar pair
    most_similar_pair = pairwise_distances.iloc[0]
    most_distinct_pair = pairwise_distances.iloc[-1]
    
    # Tightest and loosest clusters
    tightest = intra_variance.iloc[0]
    loosest = intra_variance.iloc[-1]
    
    summary = {
        'research_question': 'RQ6: Semantic Separability of Cultural Narratives',
        'main_question': 'Do embeddings cluster by culture or collapse into generic groups?',
        'total_stories': len(df),
        'num_cultures': df['culture'].nunique(),
        'num_models': df['model'].nunique(),
        'embedding_dim': len(df['story_embedding'].iloc[0]),
        
        # Overall separability
        'overall_silhouette_score': overall_silhouette,
        'silhouette_interpretation': 'Poor' if overall_silhouette < 0.1 else 'Weak' if overall_silhouette < 0.25 else 'Fair' if overall_silhouette < 0.5 else 'Good',
        
        # Best/worst separated cultures
        'best_separated_culture': best_separated['culture'],
        'best_separated_silhouette': best_separated['silhouette_mean'],
        'most_overlapping_culture': most_overlapping['culture'],
        'most_overlapping_silhouette': most_overlapping['silhouette_mean'],
        
        # Pairwise relationships
        'most_similar_pair': f"{most_similar_pair['culture_1']} - {most_similar_pair['culture_2']}",
        'most_similar_distance': most_similar_pair['cosine_distance'],
        'most_distinct_pair': f"{most_distinct_pair['culture_1']} - {most_distinct_pair['culture_2']}",
        'most_distinct_distance': most_distinct_pair['cosine_distance'],
        
        # Cluster tightness
        'tightest_cluster': tightest['culture'],
        'tightest_variance': tightest['intra_cluster_variance'],
        'loosest_cluster': loosest['culture'],
        'loosest_variance': loosest['intra_cluster_variance']
    }
    
    return summary


def main():
    """Main execution function for RQ6 analysis."""
    print("=" * 70)
    print("RQ6: Semantic Separability of Cultural Narratives")
    print("=" * 70)
    print("\nQuestion: Do embeddings cluster by culture or collapse into generic groups?")
    print("Method: UMAP/t-SNE visualization + Silhouette analysis")
    print("-" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading embedding data...")
    df = load_parquet_data('openai_text-embedding-3-large')
    print(f"    Loaded {len(df)} stories")
    print(f"    Cultures: {df['culture'].nunique()}")
    print(f"    Models: {df['model'].nunique()}")
    
    # Extract embeddings
    print("\n[2] Extracting embeddings...")
    embeddings = extract_embeddings(df)
    print(f"    Embedding shape: {embeddings.shape}")
    print(f"    Embedding dimension: {embeddings.shape[1]}")
    
    cultures = df['culture'].values
    
    # Calculate silhouette scores
    print("\n[3] Calculating silhouette scores...")
    overall_silhouette, sample_silhouettes = calculate_silhouette_scores(embeddings, cultures)
    print(f"    Overall Silhouette Score: {overall_silhouette:.4f}")
    
    # Interpretation
    if overall_silhouette < 0.1:
        interp = "POOR - Cultures largely overlap in embedding space"
    elif overall_silhouette < 0.25:
        interp = "WEAK - Some separation but significant overlap"
    elif overall_silhouette < 0.5:
        interp = "FAIR - Moderate separation between cultures"
    else:
        interp = "GOOD - Clear separation between cultural clusters"
    print(f"    Interpretation: {interp}")
    
    # Per-culture silhouette
    print("\n[4] Calculating per-culture silhouette scores...")
    per_culture_sil = calculate_per_culture_silhouette(embeddings, cultures)
    per_culture_sil.to_csv(output_dir / 'rq6_silhouette_by_culture.csv', index=False)
    print(f"    Exported: rq6_silhouette_by_culture.csv")
    
    print("\n    Per-Culture Silhouette Scores:")
    print("    " + "-" * 60)
    for _, row in per_culture_sil.iterrows():
        indicator = "✓" if row['silhouette_mean'] > 0 else "✗"
        print(f"    {indicator} {row['culture']:<25} {row['silhouette_mean']:>8.4f} (n={int(row['n'])})")
    
    # Pairwise culture distances
    print("\n[5] Calculating pairwise culture distances...")
    pairwise_distances = calculate_pairwise_culture_distances(embeddings, cultures)
    pairwise_distances.to_csv(output_dir / 'rq6_pairwise_distances.csv', index=False)
    print(f"    Exported: rq6_pairwise_distances.csv")
    
    # Distance matrix
    unique_cultures = df['culture'].unique()
    distance_matrix = create_distance_matrix(pairwise_distances, unique_cultures)
    distance_matrix.to_csv(output_dir / 'rq6_distance_matrix.csv')
    print(f"    Exported: rq6_distance_matrix.csv")
    
    # Identify overlapping cultures
    overlapping = identify_overlapping_cultures(pairwise_distances)
    overlapping.to_csv(output_dir / 'rq6_overlapping_cultures.csv', index=False)
    print(f"    Exported: rq6_overlapping_cultures.csv")
    
    print("\n    Most Similar Culture Pairs (potential overlap):")
    for _, row in overlapping.head(5).iterrows():
        print(f"      {row['culture_1']:<15} - {row['culture_2']:<15} (dist: {row['cosine_distance']:.4f})")
    
    # Distinct cultures
    distinct = identify_distinct_cultures(pairwise_distances)
    distinct.to_csv(output_dir / 'rq6_distinct_cultures.csv', index=False)
    print(f"    Exported: rq6_distinct_cultures.csv")
    
    print("\n    Most Distinct Culture Pairs:")
    for _, row in distinct.head(5).iterrows():
        print(f"      {row['culture_1']:<15} - {row['culture_2']:<15} (dist: {row['cosine_distance']:.4f})")
    
    # Intra-culture variance
    print("\n[6] Calculating intra-culture variance...")
    intra_variance = calculate_intra_culture_variance(embeddings, cultures)
    intra_variance.to_csv(output_dir / 'rq6_intra_culture_variance.csv', index=False)
    print(f"    Exported: rq6_intra_culture_variance.csv")
    
    # Dimensionality reduction
    print("\n[7] Performing dimensionality reduction...")
    
    tsne_2d = None
    umap_2d = None
    
    if HAS_TSNE:
        print("    Running t-SNE (this may take a moment)...")
        tsne_2d = perform_tsne(embeddings)
        
        # Save t-SNE coordinates
        tsne_df = pd.DataFrame({
            'story_id': range(len(df)),
            'culture': cultures,
            'model': df['model'].values,
            'tsne_1': tsne_2d[:, 0],
            'tsne_2': tsne_2d[:, 1]
        })
        tsne_df.to_csv(output_dir / 'rq6_tsne_coordinates.csv', index=False)
        print(f"    Exported: rq6_tsne_coordinates.csv")
    
    if HAS_UMAP:
        print("    Running UMAP...")
        umap_2d = perform_umap(embeddings)
        
        # Save UMAP coordinates
        umap_df = pd.DataFrame({
            'story_id': range(len(df)),
            'culture': cultures,
            'model': df['model'].values,
            'umap_1': umap_2d[:, 0],
            'umap_2': umap_2d[:, 1]
        })
        umap_df.to_csv(output_dir / 'rq6_umap_coordinates.csv', index=False)
        print(f"    Exported: rq6_umap_coordinates.csv")
    
    # Create visualizations
    print("\n[8] Creating visualizations...")
    create_visualizations(df, embeddings, tsne_2d, umap_2d, per_culture_sil, distance_matrix, output_dir)
    
    # Generate summary
    print("\n[9] Generating summary...")
    summary = generate_summary(df, overall_silhouette, per_culture_sil, pairwise_distances, intra_variance)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'rq6_summary.csv', index=False)
    print(f"    Exported: rq6_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RQ6 ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nKey Findings:")
    print(f"\n  Overall Silhouette Score: {overall_silhouette:.4f}")
    print(f"  Interpretation: {interp}")
    
    if overall_silhouette < 0.25:
        print("\n  ⚠️  WARNING: Cultural narratives show significant semantic overlap!")
        print("     Embeddings do NOT strongly separate cultures.")
    else:
        print("\n  ✓  Cultural narratives show reasonable semantic separation.")
    
    print(f"\n  Best Separated Culture:   {per_culture_sil.iloc[0]['culture']} ({per_culture_sil.iloc[0]['silhouette_mean']:.4f})")
    print(f"  Most Overlapping Culture: {per_culture_sil.iloc[-1]['culture']} ({per_culture_sil.iloc[-1]['silhouette_mean']:.4f})")
    
    print(f"\n  Most Similar Pair:  {pairwise_distances.iloc[0]['culture_1']} - {pairwise_distances.iloc[0]['culture_2']} ({pairwise_distances.iloc[0]['cosine_distance']:.4f})")
    print(f"  Most Distinct Pair: {pairwise_distances.iloc[-1]['culture_1']} - {pairwise_distances.iloc[-1]['culture_2']} ({pairwise_distances.iloc[-1]['cosine_distance']:.4f})")
    
    print("\n" + "-" * 70)
    print("Exported CSV files:")
    print("  1. rq6_silhouette_by_culture.csv - Per-culture silhouette scores")
    print("  2. rq6_pairwise_distances.csv - Pairwise culture distances")
    print("  3. rq6_distance_matrix.csv - Distance matrix")
    print("  4. rq6_overlapping_cultures.csv - Most similar culture pairs")
    print("  5. rq6_distinct_cultures.csv - Most distinct culture pairs")
    print("  6. rq6_intra_culture_variance.csv - Within-culture variance")
    print("  7. rq6_tsne_coordinates.csv - t-SNE 2D coordinates")
    print("  8. rq6_umap_coordinates.csv - UMAP 2D coordinates")
    print("  9. rq6_summary.csv - Key findings summary")
    print("-" * 70)
    
    return df, embeddings, overall_silhouette, per_culture_sil, pairwise_distances


if __name__ == "__main__":
    df, embeddings, overall_silhouette, per_culture_sil, pairwise_distances = main()
