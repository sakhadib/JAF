# RQ6: Semantic Separability of Cultural Narratives

## Research Question

**Do LLM-generated folklore stories form semantically distinct clusters by culture, or do they collapse into generic, culturally undifferentiated narrative groups?**

---

## Executive Summary

**Finding:** Cultural narratives show **POOR semantic separability**. The overall silhouette score of -0.0016 indicates that stories cluster very weakly by culture—LLMs generate narratives that are semantically similar across cultures rather than preserving distinctive cultural fingerprints.

**Implication:** AI-generated folklore tends toward cultural homogenization, with only minor semantic variation between ethnic groups' stories.

---

## Methodology

### Embedding Analysis
- **Embedding Model:** OpenAI text-embedding-3-large (3072 dimensions)
- **Stories Analyzed:** 599 complete folklore narratives
- **Cultures:** 12 Bangladeshi ethnic groups
- **Dimensionality Reduction:** t-SNE and UMAP for visualization

### Clustering Metrics
- **Silhouette Score:** Measures how similar stories are to their own culture vs. other cultures (-1 to +1)
- **Cosine Distance:** Semantic distance between culture centroids
- **Intra-cluster Variance:** Coherence within each culture's stories

### Interpretation Guide
| Silhouette Score | Interpretation |
|:----------------:|----------------|
| 0.7 to 1.0 | Strong clusters |
| 0.5 to 0.7 | Reasonable clusters |
| 0.25 to 0.5 | Weak clusters |
| < 0.25 | Poor/no clusters |
| < 0 | Wrong assignments |

---

## Results

### 1. Overall Clustering Quality

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| **Overall Silhouette** | **-0.0016** | Poor (near-random) |
| Best Culture Silhouette | 0.017 | Barely positive |
| Worst Culture Silhouette | -0.028 | Negative (overlap) |
| Number of Cultures | 12 | — |
| Embedding Dimensions | 3072 | — |

**Critical Finding:** A silhouette score near zero indicates that culture membership provides **almost no predictive power** for story semantics.

### 2. Silhouette Scores by Culture

| Rank | Culture | Silhouette | SD | Interpretation |
|:----:|---------|:----------:|:--:|----------------|
| 1 | Rakhine | **+0.017** | 0.051 | 🟡 Weakly distinct |
| 2 | Hajong | +0.012 | 0.042 | 🟡 Weakly distinct |
| 3 | Khasi | +0.011 | 0.037 | 🟡 Weakly distinct |
| 4 | Oraon (Kurukh) | +0.008 | 0.033 | 🟡 Barely distinct |
| 5 | Manipuri (Meitei) | +0.006 | 0.035 | 🟡 Barely distinct |
| 6 | Marma | +0.000 | 0.019 | ⚪ Indistinct |
| 7 | Tripura | -0.004 | 0.039 | 🟠 Slight overlap |
| 8 | Mro | -0.006 | 0.020 | 🟠 Slight overlap |
| 9 | Chakma | -0.009 | 0.026 | 🟠 Slight overlap |
| 10 | Garo (Mandi) | -0.012 | 0.028 | 🟠 Overlap |
| 11 | Santal | -0.013 | 0.024 | 🟠 Overlap |
| 12 | Bengali | **-0.028** | 0.019 | 🔴 Strong overlap |

**Key Insight:** Even the "best" culture (Rakhine at +0.017) has a silhouette score barely above zero—no culture is truly semantically distinct.

### 3. Most Similar Culture Pairs (Semantic Overlap)

| Culture 1 | Culture 2 | Cosine Distance | Relationship |
|-----------|-----------|:---------------:|--------------|
| **Marma** | **Mro** | **0.025** | 🔴 Near-identical |
| Oraon (Kurukh) | Santal | 0.040 | 🔴 Very similar |
| Chakma | Marma | 0.041 | 🔴 Very similar |
| Chakma | Garo (Mandi) | 0.043 | 🔴 Very similar |
| Chakma | Tripura | 0.045 | 🟠 Similar |
| Bengali | Chakma | 0.046 | 🟠 Similar |
| Chakma | Oraon (Kurukh) | 0.047 | 🟠 Similar |

**Interpretation:** Marma and Mro stories are semantically almost indistinguishable (distance = 0.025). Multiple culture pairs show distances under 0.05, indicating high semantic homogeneity.

### 4. Most Distinct Culture Pairs

| Culture 1 | Culture 2 | Cosine Distance | Relationship |
|-----------|-----------|:---------------:|--------------|
| **Rakhine** | **Tripura** | **0.098** | 🟢 Most distinct |
| Khasi | Rakhine | 0.092 | 🟢 Distinct |
| Oraon (Kurukh) | Rakhine | 0.090 | 🟢 Distinct |
| Mro | Rakhine | 0.089 | 🟢 Distinct |
| Rakhine | Santal | 0.087 | 🟡 Moderately distinct |

**Pattern:** Rakhine appears in all top distinct pairs—it is the most semantically unique culture. However, even the maximum distance (0.098) is quite small in absolute terms.

### 5. Intra-Culture Variance (Story Consistency)

| Rank | Culture | Variance | Interpretation |
|:----:|---------|:--------:|----------------|
| 1 | Hajong | **0.0014** | 🟢 Tightest (most consistent) |
| 2 | Oraon (Kurukh) | 0.0023 | 🟢 Consistent |
| 3 | Bengali | 0.0025 | 🟢 Consistent |
| 4 | Chakma | 0.0030 | 🟡 Moderate |
| 5 | Rakhine | 0.0030 | 🟡 Moderate |
| ... | ... | ... | ... |
| 12 | Manipuri (Meitei) | **0.0044** | 🟠 Loosest (most varied) |

**Insight:** Lower variance means stories within a culture are more similar to each other. Hajong has the tightest cluster, while Manipuri shows the most internal diversity.

### 6. Distance Ratio Analysis

| Comparison | Value |
|------------|:-----:|
| Mean inter-culture distance | 0.064 |
| Mean intra-culture distance | ~0.052 |
| Distance ratio | ~1.23 |

A ratio near 1.0 confirms that stories are almost as similar **across** cultures as they are **within** cultures—minimal cultural differentiation.

### 7. Cultural Distance Matrix (Abridged)

Top left corner of the full 12×12 distance matrix:

|  | Hajong | Santal | Tripura | Manipuri | Oraon |
|--|:------:|:------:|:-------:|:--------:|:-----:|
| Hajong | 0 | 0.054 | 0.080 | 0.076 | 0.068 |
| Santal | 0.054 | 0 | 0.075 | 0.077 | 0.040 |
| Tripura | 0.080 | 0.075 | 0 | 0.082 | 0.064 |
| Manipuri | 0.076 | 0.077 | 0.082 | 0 | 0.073 |
| Oraon | 0.068 | 0.040 | 0.064 | 0.073 | 0 |

---

## Key Findings

### 1. Cultural Collapse Confirmed

The near-zero overall silhouette (-0.0016) proves that **LLMs do not preserve cultural distinctiveness** in generated narratives.

| Evidence | Implication |
|----------|-------------|
| Silhouette ≈ 0 | Random assignment would perform equally |
| 6/12 cultures negative | Half overlap with other cultures |
| Max distance = 0.098 | Even most distinct pair is similar |

### 2. Semantic Homogenization

LLMs appear to generate a "generic folklore template" that is applied uniformly across cultures:

```
Cultural Input → LLM → Generic Narrative Structure + Minor Cultural Markers
```

### 3. Best Separated: Rakhine

Rakhine stands out as the most semantically unique culture:
- Highest silhouette (+0.017)
- Appears in all "most distinct" pairs
- Maximum distance to Tripura (0.098)

Possible explanation: Rakhine (Arakanese Buddhist) folklore may have narrative structures that differ from the predominantly Bengali/tribal patterns in training data.

### 4. Most Overlapping: Bengali

Bengali shows the most overlap with other cultures:
- Lowest silhouette (-0.028)
- Close to Chakma (0.046), Santal (0.049), Oraon (0.051)

Possible explanation: Bengali is the majority culture with the most training data—LLMs may map other cultures toward Bengali narrative patterns.

### 5. Ethnic Similarity Patterns

Geographic and linguistic proximity correlate with semantic similarity:

| Close Pairs (Low Distance) | Likely Reason |
|---------------------------|---------------|
| Marma ↔ Mro (0.025) | Both Tibeto-Burman, CHT region |
| Oraon ↔ Santal (0.040) | Both Austroasiatic, plains region |
| Chakma ↔ Marma (0.041) | Both CHT, similar contact history |

---

## Discussion

### Why Cultural Collapse Occurs

Several factors explain this disappointing finding:

1. **Training Data Dominance:** Bengali/mainstream narratives dominate LLM training
2. **Structural Similarity:** All folklore shares universal narrative elements
3. **Prompt Limitations:** Explicit cultural prompts may add surface markers but not deep structure
4. **Embedding Space:** 3072 dimensions may not capture cultural nuance

### Implications for Cultural Preservation

This finding raises concerns for AI-assisted cultural documentation:

| Risk | Severity |
|------|----------|
| Generic "folklore" replacing authentic traditions | High |
| Minority cultures mapped to majority patterns | High |
| Loss of narrative distinctiveness | Moderate |
| Homogenization of oral traditions | Moderate |

### Comparison to Human-Created Folklore

Human folklorists would expect clear separation between cultures based on:
- Unique supernatural beings
- Culture-specific moral frameworks
- Regional linguistic patterns
- Distinct ritual/ceremonial elements

The LLM's failure to reproduce this separation suggests it lacks deep cultural understanding.

---

## Practical Implications

### For Researchers
- **Embedding analysis reveals cultural collapse:** Use as diagnostic for AI folklore quality
- **Silhouette thresholds:** Accept only cultures with silhouette > 0.1 as "distinct"
- **Validate with experts:** Human folklorists should verify cultural authenticity

### For Practitioners
- **Post-generation verification required:** AI folklore needs cultural expert review
- **Enhance cultural markers:** Explicit prompting for culture-specific elements
- **Consider fine-tuning:** Culture-specific models may preserve distinctiveness

### For Cultural Organizations
- **AI limitations documented:** Current LLMs unsuitable for authentic cultural preservation
- **Hybrid approaches needed:** AI + human collaboration required
- **Training data gaps:** Need more diverse, authenticated cultural corpora

---

## Visualizations

### t-SNE Projection
The t-SNE plot (`rq6_tsne_by_culture.png`) shows cultural points largely intermixed, with no clear clusters.

### UMAP Projection
UMAP (`rq6_umap_by_culture.png`) similarly fails to reveal cultural boundaries.

### Distance Heatmap
The heatmap (`rq6_distance_heatmap.png`) shows uniformly low distances across all culture pairs.

---

## Limitations

1. **Embedding Model:** Results may differ with other embedding architectures
2. **Story Length:** Variable story lengths may affect embedding quality
3. **Cultural Ground Truth:** No authenticated human folklore corpus for comparison
4. **Single Domain:** Results specific to Bangladeshi folklore
5. **Silhouette Sensitivity:** Small sample per culture (n=50) limits precision

---

## Conclusion

**Cultural narratives generated by LLMs show POOR semantic separability.** The overall silhouette score of -0.0016 indicates that stories do not cluster meaningfully by culture—AI-generated folklore tends toward cultural homogenization.

**Key Implications:**
1. LLMs produce a "generic folklore template" applied across cultures
2. Only Rakhine shows weak cultural distinctiveness (silhouette = +0.017)
3. Bengali (majority culture) shows the strongest overlap with others
4. Geographic/linguistic neighbors produce near-identical stories (Marma-Mro: 0.025)

**For cultural preservation applications, current LLMs are insufficient.** They fail to capture the deep semantic differences that distinguish authentic ethnic folklore traditions.

---

## Statistical Artifacts

| File | Description |
|------|-------------|
| `rq6_silhouette_by_culture.csv` | Per-culture silhouette scores |
| `rq6_distinct_cultures.csv` | Most distinct culture pairs |
| `rq6_overlapping_cultures.csv` | Most similar culture pairs |
| `rq6_pairwise_distances.csv` | All pairwise distances |
| `rq6_distance_matrix.csv` | Full 12×12 distance matrix |
| `rq6_intra_culture_variance.csv` | Within-culture consistency |
| `rq6_tsne_coordinates.csv` | t-SNE projection coordinates |
| `rq6_umap_coordinates.csv` | UMAP projection coordinates |
| `rq6_summary.csv` | Executive summary data |
| `rq6_tsne_by_culture.png` | t-SNE visualization |
| `rq6_umap_by_culture.png` | UMAP visualization |
| `rq6_silhouette_by_culture.png` | Silhouette comparison |
| `rq6_distance_heatmap.png` | Pairwise distance heatmap |
| `rq6_embedding_by_model.png` | Model-colored embedding plot |

---

*Analysis conducted: January 2026*  
*Dataset: Bangladeshi Folklore Story Generation Evaluation (n=599)*  
*Embedding Model: OpenAI text-embedding-3-large (3072 dimensions)*
