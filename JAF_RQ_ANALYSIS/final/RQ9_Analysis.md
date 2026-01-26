# RQ9: Detection of Model Collapse / Repetition

## Research Question

**Do certain models produce semantically repetitive outputs (model collapse)? Which models generate the most diverse vs. formulaic folklore?**

---

## Executive Summary

**Finding:** Models differ **significantly** in repetitiveness (F = 3395, p < 0.001, η² = 0.28). **GPT-5.1** is the most repetitive (similarity = 0.525), while **Qwen-3-8B** is the most diverse (similarity = 0.427). Ironically, the highest-quality model (GPT-5.1) shows the most semantic uniformity.

**Interpretation:** GPT-5.1's "repetitiveness" may actually reflect consistent high-quality patterns rather than collapse, while Qwen's "diversity" may include noise.

---

## Methodology

### Repetitiveness Measurement
For each model, we compute pairwise semantic similarity between all stories:
- **High similarity** = Repetitive/formulaic outputs
- **Low similarity** = Diverse/varied outputs

### Metrics
- **Mean Similarity:** Average cosine similarity between story pairs (0-1 scale)
- **Diversity Score:** Mean distance from centroid (inverse of clustering)
- **Embedding Variance:** Spread in embedding space

### Statistical Tests
- One-way ANOVA (model effect on similarity)
- Effect size (η²)

---

## Results

### 1. Model Repetitiveness Ranking

| Rank | Model | Mean Similarity | SD | Interpretation |
|:----:|-------|:---------------:|:--:|----------------|
| 1 | **GPT-5.1** | **0.525** | 0.058 | 🔴 Most Repetitive |
| 2 | GPT-5-mini | 0.518 | 0.066 | 🔴 High Repetition |
| 3 | Gemini | 0.500 | 0.069 | 🟡 Moderate |
| 4 | Mistral | 0.430 | 0.077 | 🟢 Low Repetition |
| 5 | **Qwen** | **0.427** | 0.073 | 🟢 Most Diverse |

**Repetitiveness Gap:** 0.097 (9.7% difference between most and least repetitive)

### 2. ANOVA Results

| Statistic | Value | Interpretation |
|-----------|:-----:|----------------|
| F(4, ~35,000) | **3395.28** | Extremely significant |
| p-value | **< 0.001** | Highly significant |
| η² | **0.276** | Large effect size |

**Interpretation:** Model explains **27.6%** of variance in story similarity—a substantial effect.

### 3. Diversity Metrics (Inverse of Repetitiveness)

| Rank | Model | Diversity Score | Embedding Variance | Interpretation |
|:----:|-------|:---------------:|:------------------:|----------------|
| 1 | **Qwen** | **0.752** | 0.00020 | 🟢 Most Diverse |
| 2 | Mistral | 0.750 | 0.00020 | 🟢 High Diversity |
| 3 | Gemini | 0.703 | 0.00020 | 🟡 Moderate |
| 4 | GPT-5-mini | 0.690 | 0.00020 | 🟠 Low Diversity |
| 5 | **GPT-5.1** | **0.685** | 0.00020 | 🔴 Least Diverse |

### 4. Similarity Distributions

| Model | Min Sim | Median | Max Sim | Range |
|-------|:-------:|:------:|:-------:|:-----:|
| GPT-5.1 | 0.314 | 0.524 | 0.761 | 0.447 |
| GPT-5-mini | 0.306 | 0.515 | 0.775 | 0.469 |
| Gemini | 0.287 | 0.498 | 0.776 | 0.489 |
| Mistral | 0.179 | 0.426 | 0.833 | 0.653 |
| Qwen | 0.189 | 0.424 | 0.713 | 0.524 |

**Key Observation:** GPT-5.1 has the narrowest range (0.447), confirming consistent but uniform outputs.

### 5. Repetitiveness by Culture

**Most Repetitive Combinations (GPT-5.1):**

| Culture | Similarity | Interpretation |
|---------|:----------:|----------------|
| Marma | 0.574 | Very repetitive |
| Tripura | 0.572 | Very repetitive |
| Manipuri | 0.571 | Very repetitive |
| Khasi | 0.566 | Repetitive |
| Chakma | 0.561 | Repetitive |

**Most Diverse Combinations (Qwen):**

| Culture | Similarity | Interpretation |
|---------|:----------:|----------------|
| Bengali | 0.399 | Diverse |
| Manipuri | 0.403 | Diverse |
| Khasi | 0.419 | Varied |
| Garo | 0.425 | Varied |
| Oraon | 0.434 | Moderate |

### 6. The Quality-Repetitiveness Paradox

| Model | Quality Rank (RQ1) | Repetitiveness Rank | Pattern |
|-------|:------------------:|:-------------------:|---------|
| GPT-5.1 | #1 Best | #1 Most Repetitive | High quality = uniform |
| Qwen | #5 Worst | #5 Most Diverse | Low quality = varied |

**Interpretation:** The best model may have learned a "winning formula" it applies consistently, while weaker models produce more varied (but lower quality) outputs.

---

## Key Findings

### 1. Significant Model Differences

The massive F-statistic (3395) and large effect size (η² = 0.28) confirm that models differ substantially in output diversity.

### 2. OpenAI Models Most Repetitive

| Family | Mean Similarity |
|--------|:---------------:|
| OpenAI (GPT-5.1, mini) | 0.521 |
| Google (Gemini) | 0.500 |
| Others (Mistral, Qwen) | 0.429 |

OpenAI models show ~9% higher similarity than competitors.

### 3. Quality-Diversity Trade-off

```
High Quality + Repetitive = GPT-5.1 (consistent excellence)
Low Quality + Diverse = Qwen (inconsistent variation)
```

This suggests a **Pareto frontier** where optimizing one dimension may sacrifice the other.

### 4. No Model Collapse Detected

| Threshold | Interpretation | Any Models? |
|-----------|----------------|:-----------:|
| Similarity > 0.8 | Severe collapse | ❌ No |
| Similarity > 0.7 | Moderate collapse | ❌ No |
| Similarity > 0.6 | Mild uniformity | ❌ No |
| Similarity > 0.5 | Normal range | GPT-5.1, mini, Gemini |

**Verdict:** No model shows true "collapse" (extreme repetition). All operate within normal generative ranges.

---

## Discussion

### What Does "Repetitiveness" Mean for Quality?

**Two Interpretations:**

1. **Negative (Collapse):** 
   - Model has limited generative capacity
   - Produces formulaic, templated outputs
   - Lacks creativity

2. **Positive (Consistency):**
   - Model has learned effective narrative patterns
   - Applies proven structures reliably
   - Maintains quality standards

**Evidence favors the positive interpretation for GPT-5.1:**
- Highest quality scores (RQ1)
- Best cultural accuracy
- Superior narrative coherence

### Why Are OpenAI Models More Repetitive?

1. **RLHF Alignment:** Extensive fine-tuning may narrow the output distribution
2. **Style Consistency:** Training for consistent tone reduces variation
3. **Quality Threshold:** May reject diverse but low-quality outputs
4. **Prompt Sensitivity:** Less prone to prompt perturbations

### Diversity as Noise

Qwen's high diversity may partially reflect:
- Inconsistent quality across outputs
- Higher sensitivity to random seeds
- Less robust pattern learning
- More exploration, less exploitation

---

## Practical Implications

### For Researchers
- **Repetitiveness ≠ Quality:** High similarity can indicate consistency, not collapse
- **Report both metrics:** Quality and diversity are complementary
- **Context-dependent interpretation:** Repetition may be desirable for some applications

### For Practitioners
- **Need consistency?** Use GPT-5.1 (reliable quality)
- **Need variety?** Use Qwen or Mistral (but accept quality variance)
- **Creative applications:** Consider ensemble approaches

### For Developers
- **Monitor for collapse:** Track similarity over time/scale
- **Diversity injection:** Add temperature variation for more varied outputs
- **Quality-diversity balance:** Fine-tune for optimal trade-off

---

## Quality-Repetitiveness Matrix

| | Low Repetition | High Repetition |
|--|:--------------:|:---------------:|
| **High Quality** | 🎯 Ideal (rare) | GPT-5.1 ✓ |
| **Low Quality** | Qwen, Mistral | Avoid ✗ |

GPT-5.1 occupies the "High Quality + High Repetition" quadrant—a reasonable trade-off.

---

## Limitations

1. **Similarity ≠ Plagiarism:** High similarity doesn't mean identical text
2. **Embedding Compression:** Semantic similarity may miss surface variation
3. **Cultural Context:** Some repetition may be genre-appropriate for folklore
4. **Single Prompt Template:** Results may differ with varied prompts

---

## Conclusion

**Models differ significantly in output repetitiveness** (η² = 0.28). GPT-5.1 is the most repetitive (similarity = 0.525), while Qwen is the most diverse (similarity = 0.427).

**Critical Insight:** The most repetitive model is also the highest quality model. This suggests that GPT-5.1's "repetitiveness" reflects **consistent application of effective narrative patterns** rather than model collapse.

**No model shows true collapse.** All similarity scores remain below 0.55, well within normal generative ranges. The observed repetitiveness is better characterized as "stylistic consistency" than "generative failure."

**Recommendation:** For folklore generation, **accept GPT-5.1's relative uniformity** as the price of consistent quality. For maximum diversity, use Qwen—but expect lower average quality.

---

## Statistical Artifacts

| File | Description |
|------|-------------|
| `rq9_model_repetitiveness.csv` | Per-model similarity statistics |
| `rq9_diversity_metrics.csv` | Diversity scores by model |
| `rq9_repetitiveness_by_culture.csv` | Model × Culture breakdown |
| `rq9_repetitiveness_by_story_type.csv` | Model × Story type breakdown |
| `rq9_anova.csv` | ANOVA results |
| `rq9_summary.csv` | Executive summary data |
| `rq9_repetitiveness_by_model.png` | Model comparison chart |
| `rq9_similarity_distributions.png` | Distribution plots |
| `rq9_diversity_vs_repetitiveness.png` | Trade-off visualization |
| `rq9_heatmap_model_culture.png` | Heatmap by culture |
| `rq9_heatmap_model_storytype.png` | Heatmap by story type |

---

*Analysis conducted: January 2026*  
*Dataset: Bangladeshi Folklore Story Generation Evaluation (n=599)*
