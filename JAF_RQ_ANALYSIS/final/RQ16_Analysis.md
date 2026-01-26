# RQ16: Regional-Cultural Consistency Analysis

## Research Question
**Are cultures within the same geographic region appropriately distinct in their generated narratives? Do models maintain cultural boundaries or conflate neighboring cultures?**

This question investigates whether LLMs can preserve distinct cultural identities when generating stories for cultures that share the same geographic region.

---

## Executive Summary

**Critical Finding:** Regional cultures show **weak-to-moderate distinctiveness**. Only 25% of culture pairs are statistically distinguishable.

| Key Metric | Value | Interpretation |
|------------|-------|----------------|
| **Multi-Culture Regions** | 2 | Chittagong Hill Tracts, Sylhet border |
| **Culture Pairs Analyzed** | 4 | 3 pairs + 1 pair |
| **Statistically Distinct Pairs** | 1/4 (25%) | Khasi vs Manipuri only |
| **Mean Inter-Culture Distance** | 0.055 | Very small separation |
| **Intra/Inter Distance Ratio** | 0.94 | Poor clustering |

**Bottom Line:** LLMs struggle to maintain clear cultural boundaries between neighboring cultures. Only Sylhet region's Khasi-Manipuri pair shows statistically significant semantic separation.

---

## Methodology

### Regional Focus
Two regions with multiple ethnic groups:

| Region | Cultures | N Stories |
|--------|----------|-----------|
| **Chittagong Hill Tracts** | Chakma, Marma, Tripura | 150 |
| **Sylhet (border areas)** | Khasi, Manipuri (Meitei) | 100 |

### Analysis Approach
1. **Intra-Culture Consistency**: How similar are stories within same culture?
2. **Inter-Culture Distinctiveness**: How different are stories between cultures in same region?
3. **Statistical Tests**: Kolmogorov-Smirnov tests for distribution differences
4. **Clustering Analysis**: Intra vs inter-region distance ratios

### Key Metric: Distinctiveness Score
```
Distinctiveness = f(centroid_distance, pairwise_distance, statistical_significance)
```
Range: 0 (no distinction) to 1 (complete separation)

---

## Results

### 1. Intra-Culture Consistency

| Region | Culture | N | Intra-Similarity | Centroid Distance | Cultural Accuracy |
|--------|---------|---|------------------|-------------------|-------------------|
| CHT | Chakma | 50 | 0.465 | 0.310 | 2.36 |
| CHT | Marma | 50 | 0.483 | 0.297 | **2.54** |
| CHT | Tripura | 50 | 0.462 | 0.313 | 2.40 |
| Sylhet | Khasi | 50 | 0.472 | 0.306 | 2.34 |
| Sylhet | Manipuri | 50 | 0.457 | 0.316 | 2.44 |

**Key Finding:** All cultures show similar intra-culture consistency (0.46-0.48), indicating comparable internal coherence.

### 2. Pairwise Culture Distances

| Region | Culture Pair | Centroid Distance | Mean Cosine | Statistically Distinct? |
|--------|--------------|-------------------|-------------|-------------------------|
| CHT | Chakma-Marma | 0.041 | 0.535 | **No** (p=0.27) |
| CHT | Chakma-Tripura | 0.045 | 0.547 | **No** (p=0.72) |
| CHT | Marma-Tripura | 0.057 | 0.545 | **No** (p=0.55) |
| **Sylhet** | **Khasi-Manipuri** | **0.078** | 0.562 | **Yes** (p<0.001) |

**Critical Finding:** In Chittagong Hill Tracts, **all three culture pairs are statistically indistinguishable** (KS tests p > 0.27).

Only **Khasi vs Manipuri** in Sylhet region shows significant semantic separation (KS = 0.48, p < 0.001).

### 3. Regional Distinctiveness Scores

| Region | Culture Pairs | Mean Distance | % Distinct | Score |
|--------|---------------|---------------|------------|-------|
| **Sylhet** | 1 | 0.078 | **100%** | **0.59** |
| Chittagong | 3 | 0.048 | 0% | 0.18 |

**Interpretation:**
- **Sylhet**: Moderate distinctiveness (score 0.59)
- **Chittagong**: Poor distinctiveness (score 0.18)

### 4. Intra vs Inter-Region Distance Analysis

| Region | Intra-Region Distance | Inter-Region Distance | Ratio | Clustering? |
|--------|----------------------|----------------------|-------|-------------|
| Chittagong | 0.048 | 0.063 | 0.76 | **Yes** (but weak) |
| Sylhet | 0.078 | 0.070 | 1.12 | **No** |

**Interpretation:**
- **Chittagong**: Slight regional clustering (stories within region more similar than across)
- **Sylhet**: No regional clustering (intra-region distance > inter-region)

Statistical tests:
- Chittagong: t = -1.72, p = 0.097 (not significant)
- Sylhet: t = 0.87, p = 0.40 (not significant)

### 5. Model-Wise Distinctiveness

| Model | CHT Distance | Sylhet Distance | Best at Distinction? |
|-------|--------------|-----------------|---------------------|
| **Gemini-3-Flash** | 0.163 | **0.234** | **Yes (Sylhet)** |
| Mistral-Large | **0.169** | 0.194 | **Yes (CHT)** |
| Qwen3-8b | 0.107 | 0.138 | Moderate |
| GPT-5.1 | 0.076 | 0.131 | Moderate |
| GPT-5-Mini | 0.070 | 0.107 | **No (lowest)** |

**Key Finding:**
- **Gemini and Mistral create most cultural distinction** (distances 0.16-0.23)
- **GPT-5-Mini creates least distinction** (distances 0.07-0.11)
- OpenAI models tend to homogenize neighboring cultures

### 6. Distinctiveness vs Quality Correlation

| Culture | Distinctiveness | Cultural Accuracy | Average Score |
|---------|-----------------|-------------------|---------------|
| Chakma | 0.048 | 2.36 | 1.90 |
| Marma | 0.048 | **2.54** | 1.96 |
| Tripura | 0.048 | 2.40 | 1.96 |
| Khasi | **0.078** | 2.34 | 1.92 |
| Manipuri | **0.078** | 2.44 | **1.99** |

**Observation:** Higher distinctiveness (Sylhet cultures) does not correspond to higher quality. The relationship is weak or absent.

---

## Key Findings

### Finding 1: Cultural Conflation is Prevalent
- 3/4 (75%) culture pairs are **statistically indistinguishable**
- Chittagong Hill Tracts cultures (Chakma, Marma, Tripura) semantically overlap
- LLMs struggle to maintain distinct cultural identities in same region

### Finding 2: Sylhet Cultures Are Exception
- Khasi vs Manipuri shows **significant separation** (KS = 0.48, p < 0.001)
- Centroid distance 0.078 vs 0.048 for Chittagong pairs
- May reflect more distinct source material or cultural traditions

### Finding 3: Model Choice Affects Distinctiveness
- **Gemini and Mistral** preserve more cultural distinction (0.16-0.23)
- **GPT-5-Mini** shows most cultural homogenization (0.07-0.11)
- GPT-5.1 is moderate despite overall quality leadership

### Finding 4: Regional Clustering is Weak
- Intra/inter ratio ≈ 0.94 (close to 1.0)
- Stories from same region are only marginally more similar
- LLMs don't strongly encode regional coherence

### Finding 5: Distinctiveness Doesn't Guarantee Quality
- No clear correlation between cultural separation and quality scores
- Marma (lowest distinction) has highest cultural accuracy (2.54)
- Suggests distinctiveness and authenticity are independent

---

## Detailed Statistical Analysis

### Kolmogorov-Smirnov Tests

| Pair | KS Statistic | p-value | Conclusion |
|------|--------------|---------|------------|
| Chakma-Marma | 0.20 | 0.272 | Not distinct |
| Chakma-Tripura | 0.14 | 0.717 | Not distinct |
| Marma-Tripura | 0.16 | 0.549 | Not distinct |
| **Khasi-Manipuri** | **0.48** | **<0.001** | **Distinct** |

### Cosine Distance Statistics

| Pair | Min | Mean | Max | Range |
|------|-----|------|-----|-------|
| Chakma-Marma | 0.29 | 0.54 | 0.76 | 0.47 |
| Chakma-Tripura | 0.17 | 0.55 | 0.79 | 0.62 |
| Marma-Tripura | 0.24 | 0.54 | 0.80 | 0.56 |
| Khasi-Manipuri | 0.29 | 0.56 | 0.81 | 0.53 |

Wide ranges (0.47-0.62) indicate high within-pair variance, contributing to non-significance.

---

## Theoretical Interpretation

### Why Chittagong Cultures Overlap

1. **Shared Geographic Context**: Close proximity → similar environmental references
2. **Training Data Conflation**: Limited distinct source material for each culture
3. **Model Generalization**: LLMs may apply "hill tribe" template broadly
4. **Linguistic Similarities**: Some shared vocabulary/concepts

### Why Sylhet Cultures Are Distinct

1. **Different Language Families**: Khasi (Austroasiatic) vs Manipuri (Sino-Tibetan)
2. **Different Religious Traditions**: Distinct ritual practices in source material
3. **More Documented Differences**: May have clearer distinctions in training data

### The Quality Paradox

- Cultural distinctiveness ≠ cultural accuracy
- Models can be **accurate within a conflated space**
- Evaluators may not penalize lack of inter-culture distinction

---

## Practical Implications

### For Cultural Preservation
1. **Risk of homogenization**: Neighboring cultures may be conflated
2. **Need for specific prompts**: Explicit cultural markers may help
3. **Model selection matters**: Gemini/Mistral better at preservation

### For Model Development
1. **Training data curation**: Need distinct exemplars for neighboring cultures
2. **Cultural encoding**: Current embeddings may not capture micro-cultural differences
3. **Evaluation metrics**: Add inter-culture distinctiveness tests

### For Research Applications
1. **Don't assume distinctiveness**: Validate cultural separation
2. **Region-specific analysis**: Some regions worse than others
3. **Multi-model verification**: Check if distinctiveness is consistent

---

## Limitations

1. **Limited Regions**: Only 2 multi-culture regions analyzed
2. **Small Sample Per Culture**: 50 stories each
3. **Embedding-Based Only**: May miss linguistic distinctiveness
4. **No Ground Truth**: Unclear what "appropriate" distinctiveness is

---

## Conclusion

**LLMs show limited ability to maintain distinct cultural identities for neighboring ethnic groups within the same region.**

Key conclusions:
1. **75% of culture pairs are statistically indistinguishable** (3/4 pairs)
2. **Only Khasi-Manipuri** shows significant semantic separation (p < 0.001)
3. **Chittagong Hill Tracts** cultures (Chakma, Marma, Tripura) are **heavily conflated**
4. **Model choice matters**: Gemini/Mistral preserve more distinction than GPT models
5. **Distinctiveness ≠ Quality**: Conflated cultures can still receive high ratings

**Key Insight:** The "cultural collapse" phenomenon identified in RQ6 manifests specifically in regional contexts. When generating stories for culturally similar groups sharing geography, LLMs apply shared templates rather than culture-specific knowledge.

**Recommendation:** For applications requiring cultural precision among neighboring groups, use Gemini or Mistral models with highly specific cultural prompts. Current LLMs are not reliable for distinguishing closely related ethnic cultures without significant prompt engineering.

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `rq16_summary.csv` | Overall regional analysis summary |
| `rq16_intra_culture_consistency.csv` | Within-culture similarity statistics |
| `rq16_pairwise_culture_distances.csv` | Between-culture distance analysis |
| `rq16_intra_vs_inter_region.csv` | Regional clustering analysis |
| `rq16_distinctiveness_scores.csv` | Regional distinctiveness metrics |
| `rq16_model_distinctiveness.csv` | Model-wise distinction preservation |
| `rq16_distinctiveness_quality.csv` | Distinctiveness vs quality correlation |
| `rq16_multi_culture_regions.csv` | Regional composition summary |
| `rq16_distance_matrix_*.png` | Culture distance matrices per region |
| `rq16_intra_culture_consistency.png` | Consistency visualization |
| `rq16_intra_vs_inter_region.png` | Clustering comparison |
| `rq16_distinctiveness_vs_quality.png` | Quality correlation scatter |
| `rq16_distinctiveness_by_model.png` | Model comparison visualization |

---

*Analysis conducted on N=250 generated stories across 2 multi-culture regions (Chittagong Hill Tracts: 3 cultures, Sylhet: 2 cultures).*
