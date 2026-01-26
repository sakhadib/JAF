# RQ1: Comparative Model Efficacy in Cultural Authenticity

## Research Question

**Which Large Language Model (LLM) yields the highest human-rated cultural authenticity for Bangladeshi folklore story generation?**

---

## Executive Summary

**Finding:** OpenAI GPT-5.1 significantly outperforms all other models in generating culturally authentic Bangladeshi folklore stories, with a mean score of 3.13 on a 5-point scale—approximately 1 point higher than the next best model.

**Statistical Confidence:** Very High (F(4,594) = 38.59, p < 0.001, η² = 0.21)

---

## Methodology

### Data Source
- **Dataset:** 599 AI-generated Bangladeshi folklore stories
- **Evaluation Metric:** "Cultural Accuracy & Authenticity" (human-rated, 0-5 scale)
- **Models Evaluated:** 5 LLMs (~120 stories each)
- **Statistical Tests:** One-way ANOVA with Tukey HSD post-hoc comparisons

### Models Compared
1. OpenAI GPT-5.1
2. OpenAI GPT-5-mini
3. Google Gemini-3-flash-preview
4. Mistral Large (2512)
5. Qwen3-8B

---

## Results

### Model Performance Ranking

| Rank | Model | Mean Score | 95% CI | SD | Median | Range |
|:----:|-------|:----------:|:------:|:--:|:------:|:-----:|
| **1** | **openai_gpt-5.1** | **3.133** | [3.02, 3.25] | 0.647 | 3.0 | 2–4 |
| 2 | google_gemini-3-flash-preview | 2.400 | [2.24, 2.56] | 0.920 | 2.0 | 0–5 |
| 3 | openai_gpt-5-mini | 2.375 | [2.27, 2.48] | 0.581 | 2.0 | 1–3 |
| 4 | mistralai_mistral-large-2512 | 2.200 | [2.10, 2.30] | 0.559 | 2.0 | 1–3 |
| 5 | qwen_qwen3-8b | 2.135 | [2.00, 2.27] | 0.747 | 2.0 | 0–4 |

### Performance Gap Analysis

| Metric | Value |
|--------|-------|
| Best Model | GPT-5.1 (M = 3.133) |
| Worst Model | Qwen3-8B (M = 2.135) |
| Absolute Gap | **0.999 points** |
| Relative Improvement | **46.8%** |

### Statistical Significance

#### ANOVA Results
```
F(4, 594) = 38.59, p < 0.001
Effect Size: η² = 0.206 (Large)
```

**Interpretation:** Model choice explains **20.6%** of the variance in cultural authenticity scores—a substantial effect in behavioral research.

#### Pairwise Comparisons (Tukey HSD)

| Comparison | Mean Diff | p-value | Significant? |
|------------|:---------:|:-------:|:------------:|
| GPT-5.1 vs Qwen3-8B | +0.999 | <0.001 | ✓*** |
| GPT-5.1 vs Mistral | +0.933 | <0.001 | ✓*** |
| GPT-5.1 vs GPT-5-mini | +0.758 | <0.001 | ✓*** |
| GPT-5.1 vs Gemini | +0.733 | <0.001 | ✓*** |
| Gemini vs Qwen3-8B | +0.266 | 0.030 | ✓* |
| Gemini vs Mistral | +0.200 | 0.180 | ns |
| Gemini vs GPT-5-mini | +0.025 | 0.999 | ns |
| GPT-5-mini vs Qwen3-8B | +0.240 | 0.064 | ns |
| GPT-5-mini vs Mistral | +0.175 | 0.304 | ns |
| Mistral vs Qwen3-8B | +0.066 | 0.952 | ns |

*Note: *** p < 0.001, * p < 0.05, ns = not significant*

---

## Key Findings

### 1. GPT-5.1 Dominance
- **GPT-5.1 significantly outperforms ALL other models** (p < 0.001 for all pairwise comparisons)
- Achieves a mean score above 3.0 (the only model to do so)
- Minimum score of 2 (never produces "poor" or "very poor" cultural content)

### 2. Model Tier Classification

Based on statistical significance patterns, models cluster into three distinct performance tiers:

| Tier | Models | Mean Range | Interpretation |
|:----:|--------|:----------:|----------------|
| **Tier 1** | GPT-5.1 | 3.13 | Superior cultural understanding |
| **Tier 2** | Gemini-3-flash | 2.40 | Intermediate (high variance) |
| **Tier 3** | GPT-5-mini, Mistral, Qwen | 2.13–2.38 | Baseline (statistically equivalent) |

### 3. Variance Patterns

| Model | SD | Interpretation |
|-------|:--:|----------------|
| Gemini-3-flash | **0.920** | Highest variance (inconsistent quality) |
| Qwen3-8B | 0.747 | High variance |
| GPT-5.1 | 0.647 | Moderate variance |
| GPT-5-mini | **0.581** | Lowest variance (consistently mediocre) |
| Mistral | 0.559 | Low variance |

**Insight:** Gemini produces the most variable outputs—sometimes achieving high scores (up to 5) but also failing completely (scores of 0). GPT-5.1 is both higher-scoring AND more consistent.

### 4. Score Distribution Characteristics

| Model | Min | Max | Observation |
|-------|:---:|:---:|-------------|
| GPT-5.1 | 2 | 4 | **Never scores below 2** (floor effect absent) |
| Gemini | 0 | 5 | **Full range** (most variable) |
| GPT-5-mini | 1 | 3 | **Ceiling at 3** (never excellent) |
| Mistral | 1 | 3 | **Ceiling at 3** (never excellent) |
| Qwen | 0 | 4 | Can fail completely |

---

## Discussion

### Why GPT-5.1 Excels

1. **Training Data Quality:** Likely exposed to more diverse, high-quality cultural content during training
2. **Model Scale:** GPT-5.1 represents OpenAI's flagship model, with presumably larger parameter count and more sophisticated cultural reasoning
3. **Consistency:** The narrow score range (2-4) suggests reliable cultural calibration

### The Gemini Paradox

Gemini shows an interesting pattern—highest variance with a full 0-5 range. This suggests:
- Capable of excellent cultural output when "aligned" with the prompt
- But prone to complete failures (hallucination or cultural misrepresentation)
- Less reliable for production use despite occasional brilliance

### Open-Source Gap

Qwen3-8B and Mistral-Large perform significantly worse than GPT-5.1, suggesting:
- Open-source models lag in culturally-specific knowledge
- Smaller model sizes may limit cultural reasoning depth
- May require fine-tuning on Bangladeshi cultural data

---

## Practical Implications

### For Researchers
- **Use GPT-5.1** for generating culturally authentic Bangladeshi folklore
- **Avoid Gemini** if consistency is required (high failure rate)
- **Consider cost-quality tradeoff:** GPT-5-mini is cheaper but ~24% less accurate

### For Practitioners
- The **~1 point gap** between GPT-5.1 and alternatives represents a meaningful quality difference
- On typical rubrics: 2.1 = "Fair" vs 3.1 = "Good"
- Budget permitting, GPT-5.1 is the recommended choice

### For Future Work
- Fine-tune open-source models (Qwen, Mistral) on Bangladeshi cultural corpora
- Investigate Gemini's failure modes to understand variance sources
- Explore ensemble approaches combining GPT-5.1's consistency with Gemini's occasional excellence

---

## Limitations

1. **Single Metric Focus:** Analysis based solely on "Cultural Accuracy & Authenticity" scores
2. **Score Subjectivity:** Human ratings may carry annotator bias
3. **Limited Score Range:** Most scores cluster in 2-3 range, limiting discrimination
4. **Temporal Snapshot:** Model capabilities may change with updates

---

## Conclusion

**OpenAI GPT-5.1 is the definitively superior model for generating culturally authentic Bangladeshi folklore stories.** It significantly outperforms all four competing models (p < 0.001) with a large effect size (η² = 0.21). The ~47% relative improvement over the worst-performing model (Qwen3-8B) represents both a statistically and practically significant advantage.

The clear recommendation for applications requiring cultural authenticity in Bangladeshi folklore generation is to use GPT-5.1, with Gemini-3-flash as a secondary option when higher variance is acceptable.

---

## Statistical Artifacts

| File | Description |
|------|-------------|
| `rq1_descriptive_stats.csv` | Per-model descriptive statistics |
| `rq1_anova_results.csv` | One-way ANOVA test results |
| `rq1_tukey_hsd.csv` | Tukey HSD pairwise comparisons |
| `rq1_summary.csv` | Key findings summary |
| `rq1_boxplot.png` | Score distributions by model |
| `rq1_violin.png` | Density distributions by model |
| `rq1_barplot.png` | Mean scores with confidence intervals |

---

*Analysis conducted: January 2026*  
*Dataset: Bangladeshi Folklore Story Generation Evaluation (n=599)*
