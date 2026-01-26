# RQ8: Embedding Model Sensitivity Comparison

## Research Question

**Does higher embedding dimensionality capture more cultural nuance? Which embedding model is most sensitive to subtle differences between cultures?**

---

## Executive Summary

**Finding:** Higher embedding dimensions **DO correlate with greater cultural sensitivity** (r = 0.90, p = 0.040). **OpenAI-Large (3072D)** is the most sensitive model, while **OpenAI-Small (1536D)** is the least sensitive despite having more dimensions than Mistral (1024D).

**Key Insight:** Dimensionality matters, but model architecture and training also play crucial roles—dimensions alone do not guarantee sensitivity.

---

## Methodology

### Embedding Models Compared

| Model | Short Name | Dimensions |
|-------|------------|:----------:|
| OpenAI text-embedding-3-large | OpenAI-Large | 3072 |
| OpenAI text-embedding-3-small | OpenAI-Small | 1536 |
| Google gemini-embedding-001 | Gemini | 3072 |
| Qwen qwen3-embedding-8b | Qwen | 4096 |
| Mistral mistral-embed-2312 | Mistral | 1024 |

### Sensitivity Metrics
- **Mean Distance:** Average cosine distance between culture pairs
- **Distance Variance:** Spread of distances (higher = more discriminating)
- **Distance Range:** Max - Min distance (higher = more separable)
- **Coefficient of Variation (CV):** Relative variability

### Approach
For each embedding model, we computed pairwise cultural distances within each story type, then analyzed which models showed the most variation in these distances.

---

## Results

### 1. Model Sensitivity Ranking

| Rank | Model | Dims | Mean Dist | Variance | Range | Sensitivity Score |
|:----:|-------|:----:|:---------:|:--------:|:-----:|:-----------------:|
| 1 | **OpenAI-Large** | 3072 | 0.0648 | **0.00020** | **0.073** | 2.00 |
| 2 | Qwen | 4096 | 0.0598 | 0.00018 | 0.063 | 2.33 |
| 3 | Gemini | 3072 | 0.0337 | 0.00011 | 0.045 | 2.67 |
| 4 | Mistral | 1024 | 0.0062 | 0.00000 | 0.008 | 3.67 |
| 5 | **OpenAI-Small** | 1536 | 0.0183 | 0.00001 | 0.018 | 4.33 |

**Sensitivity Score:** Composite of variance, range, and CV ranks (lower = more sensitive)

### 2. Dimension-Sensitivity Correlation

| Metric | Correlation (r) | p-value | Significant? |
|--------|:---------------:|:-------:|:------------:|
| Dimensions vs Variance | **0.895** | 0.040 | ✓ Yes |
| Dimensions vs Range | 0.902 | 0.037 | ✓ Yes |
| Dimensions vs Mean Dist | 0.891 | 0.042 | ✓ Yes |

**Conclusion:** Higher dimensions = More sensitivity (p < 0.05 for all correlations)

### 3. Detailed Model Profiles

#### OpenAI-Large (3072D) — Most Sensitive
| Metric | Value |
|--------|:-----:|
| Overall Mean Distance | 0.0648 |
| Overall Variance | 0.00020 |
| Overall Range | 0.0727 |
| Story-Type Mean | 0.1844 |
| Story-Type Range | 0.1891 |

**Profile:** Largest distances AND most spread—maximally discriminates between cultures.

#### Qwen (4096D) — Second Most Sensitive
| Metric | Value |
|--------|:-----:|
| Overall Mean Distance | 0.0598 |
| Overall Variance | 0.00018 |
| Overall Range | 0.0632 |
| Story-Type Mean | 0.1122 |
| Story-Type Range | 0.1409 |

**Profile:** Despite having the most dimensions (4096), slightly less sensitive than OpenAI-Large. Architecture matters.

#### Gemini (3072D) — Moderate Sensitivity
| Metric | Value |
|--------|:-----:|
| Overall Mean Distance | 0.0337 |
| Overall Variance | 0.00011 |
| Overall Range | 0.0447 |

**Profile:** Same dimensions as OpenAI-Large but half the sensitivity—training and architecture differ significantly.

#### Mistral (1024D) — Low Sensitivity
| Metric | Value |
|--------|:-----:|
| Overall Mean Distance | 0.0062 |
| Overall Variance | 0.00000 |
| Overall Range | 0.0078 |

**Profile:** Lowest dimensions, lowest sensitivity. All cultures appear nearly identical.

#### OpenAI-Small (1536D) — Paradoxically Insensitive
| Metric | Value |
|--------|:-----:|
| Overall Mean Distance | 0.0183 |
| Overall Variance | 0.00001 |
| Overall Range | 0.0180 |

**Profile:** More dimensions than Mistral but LESS sensitive—the least discriminating model overall.

### 4. Dimension vs. Sensitivity Visualization

```
Dims:    1024    1536    3072    3072    4096
Model:  Mistral  Small   Gemini  Large   Qwen
Rank:      4       5       3       1       2
         ↑       ↑       ↑       ↑       ↑
        Low    Lowest   Mid    Highest  High
```

**Key Observation:** The relationship is not perfectly monotonic—OpenAI-Small underperforms despite more dimensions than Mistral.

### 5. Story Type Distance Analysis

Distance ranges by story type for OpenAI-Large:

| Story Type | Mean Distance | Distance Spread |
|------------|:-------------:|:---------------:|
| Human–Nature Relationship | Higher | Wide |
| Trickster Figure | Higher | Wide |
| Wisdom of Elder | Lower | Narrow |
| Sacred Space | Lower | Narrow |

**Pattern:** Concrete story types show more cultural differentiation than abstract types (consistent with RQ7).

---

## Key Findings

### 1. Dimensionality-Sensitivity Correlation Confirmed

| Evidence | Value |
|----------|:-----:|
| Correlation | r = 0.90 |
| p-value | 0.040 |
| Direction | Positive |

Higher dimensions generally provide more semantic space for cultural nuances.

### 2. Architecture Trumps Raw Dimensions

| Comparison | Higher Dims | More Sensitive |
|------------|:-----------:|:--------------:|
| OpenAI-Large vs Qwen | Qwen (4096) | OpenAI-Large (3072) |
| Gemini vs OpenAI-Small | OpenAI-Small (1536) | Gemini (3072) |
| Mistral vs OpenAI-Small | OpenAI-Small (1536) | Mistral (1024) |

**Conclusion:** Model training and architecture matter more than raw dimensionality.

### 3. OpenAI-Large is Optimal for Cultural Analysis

| Dimension | OpenAI-Large | Next Best |
|-----------|:------------:|:---------:|
| Variance | 0.00020 | 0.00018 (Qwen) |
| Range | 0.0727 | 0.0632 (Qwen) |
| Mean Distance | 0.0648 | 0.0598 (Qwen) |

OpenAI-Large leads all sensitivity metrics.

### 4. OpenAI-Small Anomaly

Despite 1536 dimensions (vs Mistral's 1024), OpenAI-Small ranks LAST:
- Possible compression of the 3072D model loses discriminative power
- "Small" variant may be optimized for efficiency, not nuance
- Architecture differences may explain the paradox

---

## Discussion

### Why Does Dimensionality Correlate with Sensitivity?

1. **Representational Capacity:** More dimensions = more axes for encoding subtle differences
2. **Sparse Coding:** Cultural nuances may require dedicated dimensions
3. **Training Signal Preservation:** Higher-dim models retain more information

### Why Doesn't Dimensionality Guarantee Sensitivity?

1. **Training Data:** Models trained on generic corpora may not emphasize cultural features
2. **Distillation Effects:** Smaller models may lose cultural nuance during compression
3. **Objective Functions:** Embedding objectives (e.g., similarity prediction) may not prioritize cultural differentiation

### Implications for RQ6 (Clustering)

RQ6 used OpenAI-Large (the most sensitive model) and still found poor cultural separability. This strengthens the conclusion that:
- **LLM-generated folklore lacks cultural distinctiveness** (not an embedding artifact)
- **Even optimal embeddings cannot find structure that doesn't exist**

---

## Practical Implications

### For Researchers
- **Use OpenAI-Large** for cultural/semantic similarity analyses
- **Avoid OpenAI-Small** for nuanced comparisons despite lower cost
- **Report embedding model** as a methodological variable

### For Practitioners
- **Trade-off awareness:** High dimensions = better discrimination but higher cost
- **Task-specific selection:** General similarity tasks can use smaller models; cultural analysis needs larger ones
- **Validate sensitivity:** Test embedding models on your specific domain

### For Developers
- **Fine-tuning potential:** Sensitivity may improve with domain-specific training
- **Dimension selection:** 3072D appears optimal; 4096D shows diminishing returns
- **Model family matters:** Same-vendor models (OpenAI) differ substantially

---

## Limitations

1. **Limited Model Sample:** Only 5 embedding models compared
2. **Single Domain:** Results specific to Bangladeshi folklore
3. **Indirect Sensitivity Measure:** Variance-based metrics are proxies for true sensitivity
4. **Correlation ≠ Causation:** Dimension-sensitivity link is observational

---

## Conclusion

**Higher embedding dimensionality DOES correlate with greater cultural sensitivity** (r = 0.90, p = 0.040), but architecture and training are equally important.

**Model Rankings:**
1. **OpenAI-Large (3072D):** Most sensitive—recommended for cultural analysis
2. **Qwen (4096D):** Second best despite most dimensions
3. **Gemini (3072D):** Moderate sensitivity
4. **Mistral (1024D):** Low sensitivity (expected given low dimensions)
5. **OpenAI-Small (1536D):** Paradoxically the least sensitive

**Key Takeaway:** For cultural folklore analysis, **use OpenAI text-embedding-3-large**. The 50% dimension advantage over small variants translates to substantially better cultural discrimination.

---

## Statistical Artifacts

| File | Description |
|------|-------------|
| `rq8_model_comparison.csv` | Full model comparison data |
| `rq8_dimension_analysis.csv` | Correlation statistics |
| `rq8_distances_by_story_type.csv` | Story-type level distances |
| `rq8_distances_openai_large.csv` | OpenAI-Large distance matrix |
| `rq8_distances_openai_small.csv` | OpenAI-Small distance matrix |
| `rq8_distances_gemini.csv` | Gemini distance matrix |
| `rq8_distances_qwen.csv` | Qwen distance matrix |
| `rq8_distances_mistral.csv` | Mistral distance matrix |
| `rq8_summary.csv` | Executive summary data |
| `rq8_model_ranking.png` | Model sensitivity ranking |
| `rq8_dimensions_vs_sensitivity.png` | Correlation visualization |
| `rq8_sensitivity_metrics.png` | Metric comparison |
| `rq8_distance_distributions.png` | Distribution plots |

---

*Analysis conducted: January 2026*  
*Dataset: Bangladeshi Folklore Story Generation Evaluation (n=599)*
