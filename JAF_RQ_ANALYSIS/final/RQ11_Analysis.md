# RQ11: Automated Score Prediction from Embeddings

## Research Question
**Can embedding representations predict human evaluation quality scores?**

This question investigates whether the semantic content captured in text embeddings contains sufficient information to predict quality ratings, potentially enabling automated quality assessment without expensive human evaluation.

---

## Executive Summary

**Critical Finding:** Embeddings provide **weak but statistically significant predictive signal** for quality scores.

| Key Metric | Value | Interpretation |
|------------|-------|----------------|
| **Best R² Achieved** | 0.153 | Very weak predictive power |
| **Best Predicted Metric** | Contextual & Temporal Appropriateness | Most predictable dimension |
| **Worst Predicted Metric** | Cultural Accuracy & Authenticity | Least predictable (R²=0.088) |
| **Significant Correlations** | 4,343/12,288 (35.3%) | About 1/3 of dimensions correlate with quality |
| **Best Model** | Random Forest | Non-linear relationships dominate |

**Bottom Line:** Embeddings capture *some* quality signals but are **insufficient for production-grade automated evaluation**. The semantic space only weakly encodes human quality judgments.

---

## Methodology

### Approach
1. **Feature Extraction**: Used 3,072-dimensional embeddings from OpenAI text-embedding-3-large
2. **Regression Models**: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
3. **Target Metrics**: Four evaluation dimensions (Cultural Accuracy, Contextual Appropriateness, Narrative Coherence, Linguistic Appropriateness)
4. **Validation**: 5-fold cross-validation for robust estimates
5. **Correlation Analysis**: 12,288 dimension-metric correlation tests

### Sample Size
- **N = 599** generated stories
- **3,072 embedding dimensions** × **4 metrics** = 12,288 correlation tests

---

## Results

### 1. Model Performance Summary

| Target Metric | Best Model | Test R² | CV R² (Mean±SD) | MAE |
|---------------|------------|---------|-----------------|-----|
| **Contextual & Temporal Appropriateness** | Random Forest | **0.153** | 0.190±0.049 | 0.536 |
| **Narrative & Symbolic Coherence** | Random Forest | 0.145 | 0.144±0.031 | 0.539 |
| **Linguistic Appropriateness (Bangla)** | Random Forest | 0.142 | 0.167±0.073 | 0.503 |
| **Cultural Accuracy & Authenticity** | Random Forest | 0.088 | 0.100±0.021 | 0.593 |

**Key Observations:**
- Random Forest consistently outperforms linear models
- All linear models (Ridge, Lasso, ElasticNet) produce **negative R²** values
- The best model explains only **15.3%** of variance
- **Cultural authenticity is least predictable** from embeddings

### 2. Model Comparison Across Algorithms

| Algorithm | Cultural R² | Contextual R² | Narrative R² | Linguistic R² |
|-----------|-------------|---------------|--------------|---------------|
| Ridge Regression | -0.494 | -0.004 | -0.312 | -0.368 |
| Lasso Regression | -0.335 | 0.000 | -0.013 | -0.122 |
| ElasticNet | -0.483 | -0.083 | -0.158 | -0.277 |
| **Random Forest** | **0.088** | **0.153** | **0.145** | **0.142** |
| Gradient Boosting | -0.043 | 0.074 | 0.106 | 0.094 |

**Critical Insight:** Linear models completely fail (negative R²), indicating the embedding-quality relationship is **fundamentally non-linear**.

### 3. Correlation Analysis

| Statistic | Value |
|-----------|-------|
| Total Correlations Tested | 12,288 |
| Significant at α=0.05 | 4,343 (35.3%) |
| Correlation Range | -0.303 to +0.304 |
| Mean Absolute Correlation | 0.068 |

**Strongest Individual Correlations:**

| Rank | Metric | Dimension | Correlation | Direction |
|------|--------|-----------|-------------|-----------|
| 1 | Narrative Coherence | Dim-2266 | +0.304 | Positive |
| 2 | Contextual Appropriateness | Dim-2266 | +0.302 | Positive |
| 3 | Cultural Accuracy | Dim-2266 | +0.302 | Positive |
| 4 | Linguistic Appropriateness | Dim-1097 | -0.303 | Negative |
| 5 | Cultural Accuracy | Dim-1831 | -0.294 | Negative |

**Notable Pattern:** Dimension 2266 correlates positively with THREE metrics, suggesting a "general quality" embedding dimension.

### 4. Feature Importance (Random Forest)

**Top Predictive Dimensions:**

| Rank | Dimension | Cultural | Contextual | Narrative | Linguistic |
|------|-----------|----------|------------|-----------|------------|
| 1 | Dim-1097 | 0.039 | **0.132** | 0.072 | **0.103** |
| 2 | Dim-7 | 0.022 | 0.008 | - | - |
| 3 | Dim-186 | - | 0.009 | 0.017 | - |
| 4 | Dim-2363 | - | - | - | 0.025 |
| 5 | Dim-2266 | 0.007 | 0.007 | 0.005 | - |

**Interpretation:** Dimension 1097 is the **single most important predictor** across all metrics, explaining 10-13% of the Random Forest's predictions.

### 5. Cross-Model Analysis (By Generation Model)

| Generation Model | Best Metric | R² | Worst Metric | R² |
|------------------|-------------|----|--------------|----|
| Gemini-3-Flash | Contextual | 0.323 | Linguistic | -0.966 |
| Mistral-Large | - | All negative | Narrative | -1.453 |
| GPT-5-Mini | - | All negative | Linguistic | -0.819 |
| GPT-5.1 | - | All negative | Cultural | -1.169 |
| Qwen3-8b | Narrative | 0.187 | Linguistic | -0.512 |

**Critical Finding:** Embeddings are **most predictive for Gemini outputs** and **least predictive for GPT-5.1**. This suggests GPT-5.1's quality variations occur in semantic dimensions not captured by standard embeddings.

### 6. Cross-Culture Analysis

**Cultures with Best Embedding-Quality Alignment:**

| Culture | Best R² | Metric | Interpretation |
|---------|---------|--------|----------------|
| **Rakhine** | 0.719 | Contextual | Excellent predictability |
| **Manipuri** | 0.624 | Contextual | Strong predictability |
| **Mro** | 0.555 | Narrative | Good predictability |
| **Hajong** | 0.419 | Linguistic | Moderate predictability |

**Cultures with Worst Alignment:**

| Culture | Worst R² | Metric | Interpretation |
|---------|----------|--------|----------------|
| **Garo (Mandi)** | -1.044 | Linguistic | Completely unpredictable |
| **Manipuri** | -0.968 | Narrative | Paradoxical (best and worst!) |
| **Santal** | -0.683 | Linguistic | Poor predictability |
| **Khasi** | -0.665 | Linguistic | Poor predictability |

**Paradox Alert:** Manipuri culture shows the **best predictability for Contextual** (R²=0.624) but **worst for Narrative** (R²=-0.968). This indicates culture-specific embedding-quality relationships.

---

## Key Findings

### Finding 1: Embeddings Contain Quality Signals, But Weakly
- 35.3% of embedding dimensions show significant correlation with quality
- But correlations are weak (max |r| = 0.304)
- Only ~15% of variance explained even with best models

### Finding 2: Non-Linear Relationships Dominate
- Linear models (Ridge, Lasso, ElasticNet) all fail with negative R²
- Random Forest (non-linear) achieves positive R²
- Implies quality judgments depend on **complex interactions** of semantic features

### Finding 3: Contextual Appropriateness Most Predictable
- Temporal and contextual elements are **most embedded-predictable** (R²=0.153)
- Cultural authenticity is **least predictable** (R²=0.088)
- Suggests cultural knowledge is not well-captured in standard embeddings

### Finding 4: Single Dimension Dominates
- Dimension 1097 accounts for 10-13% of feature importance
- This dimension likely encodes a **latent quality factor**
- But its exact semantic meaning remains opaque

### Finding 5: Culture-Specific Predictability
- Some cultures (Rakhine, Manipuri) show strong R² > 0.5 for specific metrics
- Others (Garo, Santal) show negative R², meaning embeddings **mislead** predictions
- Suggests need for culture-specific embedding models

---

## Statistical Validation

### Cross-Validation Stability

| Metric | CV Mean R² | CV Std |
|--------|------------|--------|
| Contextual | 0.190 | 0.049 |
| Linguistic | 0.167 | 0.073 |
| Narrative | 0.144 | 0.031 |
| Cultural | 0.100 | 0.021 |

- CV estimates are **consistent with test R²**, indicating no overfitting
- Low standard deviations suggest **stable** (if weak) predictive signal

### Effect Size Interpretation

| R² | Effect Size | Our Results |
|----|-------------|-------------|
| 0.01 | Small | - |
| 0.09 | Medium | Cultural (0.088) |
| 0.25 | Large | None achieved |

Our best predictions reach only **medium effect size**, indicating embeddings capture some but not most quality-relevant information.

---

## Practical Implications

### For Automated Evaluation
1. **Cannot replace human evaluation**: Max R²=0.15 is insufficient for production use
2. **Useful for pre-filtering**: Could flag likely low-quality outputs for human review
3. **Confidence bounds**: Predictions have ±0.5 MAE on a ~1-5 scale

### For Embedding Model Development
1. **Cultural encodings weak**: Standard embeddings don't capture cultural authenticity well
2. **Domain-specific training needed**: Fine-tuning on cultural texts may improve predictability
3. **Non-linear decoders required**: Linear probing insufficient

### For Research
1. **Quality is not semantic**: Human quality judgments partially independent of semantic content
2. **Dimension 1097 mystery**: Investigating this dimension could reveal quality-encoding mechanisms
3. **Culture-aware embeddings needed**: Different cultures need different embedding spaces

---

## Limitations

1. **Single Embedding Model**: Only tested text-embedding-3-large; other models may perform differently
2. **Sample Size**: N=599 may be insufficient for high-dimensional regression
3. **Curse of Dimensionality**: 3,072 features vs 599 samples creates overfitting risk
4. **Cross-Validation Only**: No held-out test set from different data collection

---

## Conclusion

**Embeddings provide weak but non-trivial signal for predicting human quality ratings.**

The maximum R² of 0.153 (15.3% variance explained) demonstrates that:
1. Semantic content **partially correlates** with perceived quality
2. But **85%+ of quality judgment** comes from factors not captured in embeddings
3. **Non-linear models required** - linear probing fails completely

This suggests that human quality assessment involves **holistic, emergent properties** that transcend the semantic features captured by current embedding models. Cultural authenticity, in particular, appears to reside in knowledge structures not well-represented in standard text embeddings.

**Recommendation:** Embeddings can supplement but cannot replace human evaluation for cultural text generation. Future work should explore culture-specific embedding fine-tuning.

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `rq11_summary.csv` | Overall analysis summary |
| `rq11_model_performance.csv` | R², MAE, RMSE for all model-metric combinations |
| `rq11_dimension_correlations.csv` | 12,288 dimension-metric correlations |
| `rq11_cross_analysis.csv` | R² breakdown by generation model and culture |
| `rq11_feature_importance_*.csv` | Random Forest feature importances (4 files) |
| `rq11_by_culture.png` | Culture-wise prediction performance |
| `rq11_by_generation_model.png` | Model-wise prediction performance |
| `rq11_r2_heatmap.png` | R² heatmap across algorithms and metrics |
| `rq11_correlation_distribution.png` | Distribution of dimension correlations |

---

*Analysis conducted on N=599 generated stories with 3,072-dimensional embeddings and 4 quality metrics.*
