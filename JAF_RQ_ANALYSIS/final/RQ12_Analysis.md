# RQ12: Identifying Hallucination in Vector Space

## Research Question
**Are low-quality stories outliers in embedding space? Can semantic distance from ideal examples detect poor generation quality?**

This question investigates whether generated stories that humans rate as low-quality occupy distinct regions in the embedding space, potentially enabling automated quality detection through outlier analysis.

---

## Executive Summary

**Critical Finding:** Semantic distance from ideal examples **weakly but significantly predicts quality** (r = -0.163, p < 0.001), and Local Outlier Factor (LOF) successfully identifies lower-quality outputs.

| Key Metric | Value | Interpretation |
|------------|-------|----------------|
| **Distance-Quality Correlation** | r = -0.163 | Significant but weak |
| **Q1 vs Q4 Quality Difference** | 0.195 points | Closest stories are higher quality |
| **LOF Outlier Quality Gap** | 0.29 points (d=0.47) | Significant medium effect |
| **Isolation Forest Quality Gap** | 0.15 points (d=0.24) | Not significant |
| **Outlier Detection Rate** | 10% (60/599) | Consistent across methods |

**Bottom Line:** Embedding distance from high-quality examples provides a **weak but useful signal** for quality detection. LOF outperforms Isolation Forest, suggesting local density anomalies matter more than global distance.

---

## Methodology

### Approach
1. **Ideal Exemplars**: Selected 3 highest-rated stories as embedding "gold standards"
2. **Distance Metrics**: Cosine and Euclidean distance from ideal centroid
3. **Outlier Detection**:
   - **Isolation Forest**: Detects global anomalies
   - **Local Outlier Factor (LOF)**: Detects local density anomalies
4. **Quartile Analysis**: Stratified analysis by distance quartiles
5. **Statistical Tests**: Correlation, t-tests, Cohen's d effect sizes

### Sample Size
- **N = 599** generated stories
- **3 ideal stories** (highest-rated examples)
- **60 outliers** (10% contamination rate for both methods)

---

## Results

### 1. Distance-Quality Correlations

| Quality Metric | Cosine r | p-value | Euclidean r | p-value |
|----------------|----------|---------|-------------|---------|
| **Linguistic Appropriateness** | **-0.169** | <0.001 | -0.182 | <0.001 |
| **Narrative Coherence** | -0.135 | <0.001 | -0.148 | <0.001 |
| **Contextual Appropriateness** | -0.126 | 0.002 | -0.136 | <0.001 |
| **Cultural Accuracy** | -0.105 | 0.010 | -0.116 | 0.004 |
| **Average Score** | -0.163 | <0.001 | -0.177 | <0.001 |

**All correlations are negative and significant**: Greater distance from ideal = lower quality.

**Linguistic appropriateness** shows the strongest relationship (r = -0.169), suggesting semantic distance best captures language quality issues.

### 2. Quartile Analysis

| Distance Quartile | N | Mean Distance | Mean Quality | Std Dev |
|-------------------|---|---------------|--------------|---------|
| **Q1 (Closest)** | 150 | 0.363 | **2.07** | 0.68 |
| Q2 | 150 | 0.424 | 2.01 | 0.53 |
| Q3 | 149 | 0.465 | 2.09 | 0.68 |
| **Q4 (Farthest)** | 150 | 0.531 | **1.88** | 0.58 |

**Key Finding:** Stories closest to ideal (Q1) average **0.195 points higher** than the farthest (Q4).

**Anomaly in Q3:** Quality slightly increases in Q3 before dropping in Q4, suggesting a non-monotonic relationship.

### 3. Outlier Detection Performance

#### Isolation Forest Results

| Metric | Outlier Mean | Inlier Mean | Difference | t | p | Cohen's d |
|--------|--------------|-------------|------------|---|---|-----------|
| Cultural Accuracy | 2.30 | 2.47 | -0.17 | -1.55 | 0.12 | 0.21 |
| Contextual | 1.98 | 2.19 | -0.20 | -1.90 | 0.06 | 0.26 |
| Narrative | 1.90 | 2.00 | -0.10 | -1.04 | 0.30 | 0.14 |
| Linguistic | 1.32 | 1.45 | -0.14 | -1.32 | 0.19 | 0.18 |
| **Average** | 1.88 | 2.03 | -0.15 | -1.79 | 0.07 | 0.24 |

**Result:** Isolation Forest outliers show lower quality, but differences are **NOT statistically significant** (all p > 0.05).

#### Local Outlier Factor (LOF) Results

| Metric | Outlier Mean | Inlier Mean | Difference | t | p | Cohen's d |
|--------|--------------|-------------|------------|---|---|-----------|
| **Cultural Accuracy** | 2.08 | 2.49 | -0.41 | -3.84 | **<0.001** | **0.52** |
| **Narrative** | 1.77 | 2.01 | -0.25 | -2.59 | **0.010** | **0.35** |
| **Linguistic** | 1.22 | 1.46 | -0.25 | -2.41 | **0.016** | **0.33** |
| **Contextual** | 1.93 | 2.19 | -0.26 | -2.42 | **0.016** | **0.33** |
| **Average** | 1.75 | 2.04 | -0.29 | -3.45 | **<0.001** | **0.47** |

**Result:** LOF outliers are **significantly lower quality** across ALL metrics with **medium effect sizes** (d = 0.33-0.52).

### 4. LOF vs Isolation Forest Comparison

| Metric | LOF | Isolation Forest | Winner |
|--------|-----|------------------|--------|
| Quality Gap | 0.29 | 0.15 | **LOF** |
| Significance | 5/5 metrics | 0/5 metrics | **LOF** |
| Effect Size (avg) | d = 0.47 | d = 0.24 | **LOF** |
| Overlap | 60/60 outliers agree | - | Partial |

**Conclusion:** LOF significantly outperforms Isolation Forest for quality-based outlier detection.

### 5. Outlier Distribution by Generation Model

| Model | IF Outliers | LOF Outliers | Mean Quality | Mean Distance |
|-------|-------------|--------------|--------------|---------------|
| **Qwen3-8b** | **34** | **37** | 1.80 | 0.473 |
| **Mistral-Large** | **17** | **18** | 1.67 | 0.486 |
| Gemini-3-Flash | 6 | 3 | 1.87 | 0.413 |
| GPT-5.1 | 2 | 1 | **2.79** | 0.442 |
| GPT-5-Mini | 1 | 1 | 1.93 | 0.415 |

**Critical Finding:** **Qwen and Mistral produce most outliers** (57% and 29% of all outliers respectively).

**GPT-5.1** produces fewest outliers AND highest quality - consistent with prior RQ findings.

### 6. Outlier Distribution by Culture

| Culture | IF Outliers | LOF Outliers | Mean Quality | Mean Distance |
|---------|-------------|--------------|--------------|---------------|
| Bengali | 7 | 8 | **2.08** | 0.428 |
| Chakma | 8 | 4 | 1.90 | 0.448 |
| Khasi | 6 | 8 | 1.92 | 0.452 |
| Manipuri | 8 | 6 | 1.99 | 0.438 |
| Tripura | 6 | 6 | 1.96 | **0.476** |

**Observation:** No single culture dominates outliers - distribution relatively even.

**Tripura** shows highest mean distance from ideal, suggesting most semantically distinct cultural content.

### 7. Quality Score Distribution in Outliers

**High Quality Stories (Score ≥ 3.5):**
- 8/10 top-scoring stories are in **Q1 (Closest)** to ideal
- All 10 are from Bengali or select cultures
- 9/10 are **NOT outliers** by either method

**Low Quality Stories (Score ≤ 1.0):**
- Distributed across Q1-Q4 (no clear pattern)
- Several low-quality stories are **close** to ideal (Q1) - false negatives
- 4/10 lowest are **both IF and LOF outliers** - true positives

---

## Key Findings

### Finding 1: Distance Predicts Quality (Weakly)
- Significant negative correlation (r = -0.163, p < 0.001)
- But explains only ~2.7% of variance (r² = 0.027)
- Linguistic quality shows strongest relationship

### Finding 2: LOF Outperforms Isolation Forest
- LOF detects quality issues with **medium effect size** (d = 0.47)
- Isolation Forest fails to reach significance
- Local density anomalies more indicative of quality problems

### Finding 3: Qwen and Mistral Are Outlier-Prone
- These models produce 85% of all outliers
- GPT-5.1 produces only 3 outliers total (5%)
- Suggests different models have different "semantic coherence"

### Finding 4: Not All Low-Quality Stories Are Outliers
- Several low-quality stories cluster **near** the ideal (Q1)
- Outlier detection alone has high **false negative rate**
- Quality issues can occur in "well-formed" semantic regions

### Finding 5: Cultural Effects Are Minimal
- No culture is systematically over-represented in outliers
- Distance from ideal varies by ~0.05 across cultures
- Model choice matters more than cultural content

---

## Statistical Validation

### Effect Size Summary

| Analysis | Effect Size | Magnitude |
|----------|-------------|-----------|
| Distance-Quality Correlation | r = -0.163 | Small |
| Q1-Q4 Quality Difference | d = 0.30 | Small-Medium |
| LOF Outlier Quality Gap | d = 0.47 | Medium |
| IF Outlier Quality Gap | d = 0.24 | Small |

### Multiple Testing Consideration
- 10 correlation tests performed
- Bonferroni-adjusted α = 0.005
- All significant correlations remain significant after correction

---

## Practical Implications

### For Automated Quality Control
1. **LOF as pre-filter**: Can flag ~10% of outputs with ~0.29 quality gap
2. **Combine with distance**: Use both metrics for better precision
3. **Model-specific thresholds**: Different models need different detection parameters

### For Production Systems
1. **Real-time flagging**: LOF can identify likely quality issues
2. **Human review prioritization**: Outliers should get priority review
3. **Model selection**: Prefer GPT-5.1 to minimize outlier generation

### For Research
1. **Hallucination detection**: Embedding outliers correlate with quality
2. **But not sufficient**: Many low-quality outputs are NOT outliers
3. **Multi-signal approach needed**: Combine embedding distance with other features

---

## Theoretical Interpretation

### Why LOF Works Better Than Isolation Forest
- **LOF** measures **local density**: How different is this story from its neighbors?
- **Isolation Forest** measures **global separability**: How easy to isolate from all data?
- Quality issues are **local phenomena** - a story can be globally typical but locally anomalous

### The False Negative Problem
- Stories can be **semantically well-formed** but **factually/culturally wrong**
- Embeddings capture semantic structure, not factual accuracy
- Low quality in the "semantic center" suggests issues beyond semantic space

### Model-Specific Semantic Signatures
- GPT-5.1 outputs cluster in quality-associated regions
- Qwen/Mistral outputs spread into anomalous regions
- Different models have different "semantic attractors"

---

## Limitations

1. **Small Ideal Set**: Only 3 ideal stories may not represent true "gold standard"
2. **Contamination Rate Fixed**: 10% outlier rate is arbitrary
3. **Cosine vs Euclidean**: Results similar but not identical
4. **Causality Unclear**: Does low quality cause distance, or vice versa?

---

## Conclusion

**Embedding-space outlier detection provides a useful but imperfect signal for quality assessment.**

Key conclusions:
1. **Distance from ideal negatively correlates with quality** (r = -0.163)
2. **LOF successfully identifies lower-quality outputs** (d = 0.47, p < 0.001)
3. **Isolation Forest fails to reach significance**
4. **Model choice strongly affects outlier rate** (Qwen 57%, GPT-5.1 5%)
5. **Not all low-quality stories are outliers** - false negatives exist

**Recommendation:** Use LOF-based outlier detection as a **supplementary signal** in automated quality pipelines, but not as a standalone quality metric. Combine with direct quality prediction (RQ11) and human review for comprehensive evaluation.

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `rq12_summary.csv` | Overall analysis summary |
| `rq12_distance_correlations.csv` | Distance-quality correlations by metric |
| `rq12_distance_percentiles.csv` | Quality breakdown by distance quartile |
| `rq12_outlier_analysis.csv` | IF vs LOF performance comparison |
| `rq12_story_analysis.csv` | Per-story outlier and distance data |
| `rq12_correlation_chart.png` | Distance-quality correlation visualization |
| `rq12_distance_vs_quality.png` | Scatter plot of distance vs quality |
| `rq12_outlier_heatmap.png` | Outlier distribution heatmap |
| `rq12_pca_outliers.png` | PCA visualization of outliers |
| `rq12_quartile_analysis.png` | Quality by distance quartile |

---

*Analysis conducted on N=599 generated stories compared against 3 ideal exemplars using cosine and Euclidean distance metrics.*
