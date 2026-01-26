# RQ13: Semantic Drift vs. Human Perception

## Research Question
**Does prompt-story divergence predict coherence scores? Is there a critical drift threshold beyond which story quality degrades?**

This question investigates whether the semantic distance between a generation prompt and the resulting story ("semantic drift") correlates with human-perceived coherence and quality.

---

## Executive Summary

**Critical Finding:** Semantic drift shows **weak positive correlation** with quality (r = 0.10, p = 0.015), **contrary to expectations**.

| Key Metric | Value | Interpretation |
|------------|-------|----------------|
| **Mean Drift** | 0.642 (SD: 0.076) | High baseline divergence from prompts |
| **Drift-Quality Correlation** | r = +0.10, p = 0.015 | Weak POSITIVE (unexpected) |
| **Drift-Coherence Correlation** | r = +0.07, p = 0.088 | Not significant |
| **Lowest Drift Model** | Qwen3-8b (0.609) | Most prompt-adherent |
| **Highest Drift Model** | GPT-5.1 (0.687) | Least prompt-adherent |
| **Critical Threshold** | 0.611 | Based on decile analysis |

**Bottom Line:** Contrary to hypothesis, **higher drift does not predict lower quality**. In fact, slight positive correlation suggests more creative divergence from prompts may enhance perceived quality.

---

## Methodology

### Approach
1. **Drift Calculation**: Cosine distance between prompt embedding and story embedding
2. **Correlation Analysis**: Pearson and Spearman correlations with quality metrics
3. **Stratified Analysis**: By model, story type, and culture
4. **Threshold Detection**: Decile analysis to find quality inflection points

### Metrics
- **Drift Range**: 0 (identical) to 1 (orthogonal)
- **Observed Range**: 0.364 - 0.816

### Sample Size
- **N = 599** generated stories
- All stories generated from structured prompts

---

## Results

### 1. Overall Drift Statistics

| Statistic | Value |
|-----------|-------|
| Mean Drift | 0.642 |
| Standard Deviation | 0.076 |
| Minimum | 0.364 |
| Maximum | 0.816 |
| Median | 0.641 |

**Interpretation:** High baseline drift (0.64) indicates all models substantially transform prompt content into narratives. This is expected - stories should elaborate on prompts.

### 2. Drift-Quality Correlations

| Quality Metric | Pearson r | p-value | Spearman ρ | p-value | Significant? |
|----------------|-----------|---------|------------|---------|--------------|
| **Contextual Appropriateness** | **+0.106** | **0.009** | +0.118 | 0.004 | **Yes** |
| **Average Score** | **+0.100** | **0.015** | +0.103 | 0.012 | **Yes** |
| **Linguistic Appropriateness** | **+0.089** | **0.029** | +0.114 | 0.005 | **Yes** |
| Narrative Coherence | +0.070 | 0.088 | +0.096 | 0.019 | Mixed |
| Cultural Accuracy | +0.061 | 0.137 | +0.068 | 0.094 | No |

**Key Finding:** All correlations are **POSITIVE** - higher drift associates with **higher** quality scores.

**Strongest Effect:** Contextual appropriateness (r = +0.106) - stories that diverge more from prompts receive better temporal/contextual ratings.

### 3. Drift by Generation Model

| Model | Mean Drift | Std | Drift-Coherence r | Mean Coherence |
|-------|------------|-----|-------------------|----------------|
| **Qwen3-8b** | **0.609** | 0.083 | -0.166 (NS) | 1.78 |
| GPT-5-Mini | 0.618 | 0.075 | -0.083 (NS) | 1.87 |
| Gemini-3-Flash | 0.635 | 0.063 | -0.128 (NS) | 1.88 |
| Mistral-Large | 0.659 | 0.074 | -0.126 (NS) | 1.68 |
| **GPT-5.1** | **0.687** | 0.057 | +0.067 (NS) | **2.74** |

**Paradox Discovered:**
- **GPT-5.1 has highest drift (0.687) AND highest quality (2.74)**
- **Qwen3-8b has lowest drift (0.609) AND lower quality (1.78)**

This directly contradicts the hypothesis that drift degrades quality.

**Model-Level Correlations:** All within-model correlations are non-significant, suggesting drift-quality relationship doesn't hold within individual models.

### 4. Drift by Story Type

| Story Type | Mean Drift | Coherence r | Mean Coherence |
|------------|------------|-------------|----------------|
| **Human-Nature Relationship** | **0.535** | +0.04 (NS) | **2.03** |
| Trickster/Clever Figure | 0.598 | **+0.29** (p=0.02) | 1.95 |
| Everyday Life Narrative | 0.620 | +0.21 (NS) | **2.10** |
| Change and Continuity | 0.630 | +0.09 (NS) | 1.93 |
| Community Crisis | 0.655 | +0.01 (NS) | 2.00 |
| Rite of Passage | 0.658 | +0.21 (NS) | 2.00 |
| Moral Transgression | 0.661 | +0.14 (NS) | 1.95 |
| Origin Story | 0.672 | -0.01 (NS) | 2.08 |
| Sacred/Forbidden Space | 0.679 | +0.06 (NS) | 1.88 |
| **Wisdom of an Elder** | **0.709** | -0.02 (NS) | 1.97 |

**Observations:**
- **Human-Nature Relationship** has lowest drift - prompts most constraining
- **Wisdom of an Elder** has highest drift - prompts allow most elaboration
- Only **Trickster stories** show significant drift-coherence correlation

### 5. Drift by Culture

| Culture | Mean Drift | Coherence r | Significant? |
|---------|------------|-------------|--------------|
| Rakhine | 0.632 | +0.03 | No |
| Marma | 0.633 | +0.04 | No |
| Manipuri | 0.634 | -0.04 | No |
| Chakma | 0.635 | +0.11 | No |
| Mro | 0.638 | -0.12 | No |
| Bengali | 0.643 | -0.04 | No |
| Tripura | 0.645 | +0.23 | No |
| Oraon | 0.646 | +0.00 | No |
| Hajong | 0.647 | +0.04 | No |
| Santal | 0.648 | +0.04 | No |
| Garo | 0.649 | +0.18 | No |
| **Khasi** | 0.652 | **+0.40** | **Yes (p=0.004)** |

**Exception:** Khasi culture shows **strong positive drift-coherence correlation** (r = 0.40, p = 0.004).

This suggests for Khasi stories specifically, prompt divergence strongly predicts better coherence.

### 6. Threshold Analysis (Deciles)

| Decile | Drift Range | Mean Coherence | Change from Baseline |
|--------|-------------|----------------|---------------------|
| 1 | 0.36-0.54 | 1.95 | Baseline |
| 2 | 0.54-0.58 | 2.02 | +3.4% |
| 3 | 0.58-0.61 | 1.75 | **-10.3%** |
| 4 | 0.61-0.63 | 1.95 | 0% |
| 5 | 0.63-0.65 | 2.02 | +3.4% |
| 6 | 0.65-0.67 | 1.97 | +0.8% |
| 7 | 0.67-0.69 | 2.00 | +2.6% |
| 8 | 0.69-0.70 | **2.20** | **+12.8%** |
| 9 | 0.70-0.73 | 1.92 | -1.7% |
| 10 | 0.73-0.82 | **2.13** | +9.4% |

**No Critical Threshold Found:**
- Quality fluctuates non-monotonically with drift
- Decile 3 (drift 0.58-0.61) shows unexpected dip
- **Decile 8-10 (highest drift) show highest quality**

### 7. Drift Bins Summary

| Drift Bin | N | Mean Drift | Mean Quality |
|-----------|---|------------|--------------|
| Medium (0.3-0.4) | 2 | 0.377 | 2.25 |
| High (0.4-0.5) | 28 | 0.463 | 2.13 |
| Very High (0.5+) | 569 | 0.651 | **2.00** |

**Note:** Very few stories (30/599 = 5%) have "moderate" drift. Nearly all generation involves substantial prompt transformation.

---

## Key Findings

### Finding 1: Drift Does NOT Predict Lower Quality
- Positive correlation (r = +0.10) contradicts hypothesis
- Higher drift weakly associates with **better** quality
- Effect is small but statistically significant

### Finding 2: GPT-5.1 Paradox Explained
- GPT-5.1 has highest drift (0.687) and highest quality (2.74)
- This model creatively elaborates prompts rather than closely following
- "Prompt adherence" ≠ "quality" for this best-performing model

### Finding 3: Story Type Affects Drift More Than Culture
- Drift ranges from 0.54 (Human-Nature) to 0.71 (Wisdom of Elder)
- Cultural drift variation is minimal (0.63-0.65)
- Some prompts inherently allow more creative latitude

### Finding 4: No Critical Drift Threshold Exists
- Quality doesn't consistently degrade beyond any threshold
- Non-monotonic pattern suggests other factors dominate
- High-drift stories (deciles 8-10) have above-average quality

### Finding 5: One Culture Shows Strong Effect
- Khasi stories show r = 0.40 drift-coherence correlation
- This culture may have prompts that benefit from elaboration
- Or cultural content that emerges through divergent generation

---

## Theoretical Interpretation

### Why Higher Drift Might Mean Better Quality

1. **Creative Elaboration**: Good stories add richness beyond prompts
2. **Model Confidence**: Higher drift may indicate model "ownership" of narrative
3. **Prompt Limitations**: Constrained prompts may produce constrained stories
4. **Cultural Richness**: Cultural details require deviation from generic prompts

### The GPT-5.1 Phenomenon
GPT-5.1's combination of:
- Highest drift (least prompt-adherent)
- Highest quality (best human ratings)
- Lowest repetition (from RQ9)

Suggests this model has learned **when and how** to deviate from prompts productively.

### Implications for Prompt Engineering
1. **Tight prompt adherence may limit quality**
2. **Some prompts benefit from loose interpretation**
3. **Model capability to "know when to diverge" may be a quality signal**

---

## Statistical Validation

### Effect Sizes

| Relationship | r | R² | Effect Size |
|--------------|---|----|----|
| Drift-Quality | +0.10 | 0.01 | Small |
| Drift-Contextual | +0.11 | 0.01 | Small |
| Khasi drift-coherence | +0.40 | 0.16 | Medium-Large |

### Robustness Check
- Pearson and Spearman correlations agree in direction
- Results consistent across both parametric and non-parametric tests
- Cultural stratification reveals heterogeneity

---

## Practical Implications

### For Prompt Engineering
1. **Don't over-constrain**: Some prompt deviation is beneficial
2. **Model-specific**: GPT-5.1 benefits from creative latitude
3. **Story-type matters**: Human-Nature prompts more constraining

### For Quality Monitoring
1. **High drift ≠ hallucination**: Cannot use drift as quality proxy
2. **Reverse signal**: Very low drift may indicate template-following
3. **Cultural context**: Khasi-specific effects warrant investigation

### For Model Evaluation
1. **Prompt adherence overvalued**: RQ7 found quality-adherence trade-off
2. **This confirms**: Best model (GPT-5.1) has lowest adherence + highest drift
3. **Reframe metric**: "Creative elaboration" vs "prompt drift"

---

## Limitations

1. **Embedding-Based Drift**: May not capture semantic nuance
2. **Prompt Complexity**: Not all prompts are equally detailed
3. **Cultural Confounds**: Khasi effect may be spurious
4. **Small Effect Sizes**: Practical significance is limited

---

## Conclusion

**Semantic drift from prompts does not predict quality degradation; if anything, it weakly predicts better quality.**

This finding challenges common assumptions about prompt adherence:
1. **r = +0.10**: Small but significant positive correlation
2. **No critical threshold**: Quality doesn't degrade at high drift
3. **Best model drifts most**: GPT-5.1's success involves creative elaboration
4. **Story type determines drift**: Some narratives inherently diverge more

**Key Insight:** "Prompt adherence" should not be equated with "quality." The ability to productively elaborate on prompts - adding cultural richness, narrative depth, and contextual detail - appears to be a marker of generation quality, not a defect.

**Recommendation:** Evaluate generated content on output quality, not input adherence. High-quality models like GPT-5.1 demonstrate that knowing *when and how* to deviate from prompts is itself a capability.

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `rq13_summary.csv` | Overall drift analysis summary |
| `rq13_drift_correlations.csv` | Drift vs quality correlations by metric |
| `rq13_drift_by_model.csv` | Model-wise drift statistics |
| `rq13_drift_by_story_type.csv` | Story type drift analysis |
| `rq13_drift_by_culture.csv` | Culture-wise drift statistics |
| `rq13_threshold_analysis.csv` | Decile-based threshold detection |
| `rq13_drift_bins.csv` | Binned drift analysis |
| `rq13_drift_deciles.csv` | Detailed decile breakdown |
| `rq13_story_drift_data.csv` | Per-story drift values |
| `rq13_drift_vs_coherence.png` | Scatter plot of drift vs coherence |
| `rq13_drift_by_model.png` | Model comparison visualization |
| `rq13_drift_correlations.png` | Correlation heatmap |
| `rq13_drift_heatmap.png` | Drift distribution heatmap |
| `rq13_threshold_analysis.png` | Threshold detection visualization |

---

*Analysis conducted on N=599 generated stories measuring cosine distance between prompt and story embeddings.*
