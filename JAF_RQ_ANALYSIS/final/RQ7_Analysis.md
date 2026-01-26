# RQ7: Prompt Adherence Quantification

## Research Question

**Which generation model stays truest to the structural prompt template? How well do LLMs follow the specified story format when generating cultural folklore?**

---

## Executive Summary

**Finding:** Models vary significantly in prompt adherence (F = 23.94, p < 0.001, η² = 0.139). **Qwen-3-8B** achieves the highest structural fidelity (M = 0.391), while **GPT-5.1 shows the lowest adherence** (M = 0.313)—despite GPT-5.1 ranking #1 on subjective quality metrics.

**Key Paradox:** The model that produces the highest-quality content (GPT-5.1) is the least obedient to structural prompts, suggesting a trade-off between creativity and compliance.

---

## Methodology

### Adherence Scoring
Prompt adherence measures the semantic similarity between:
- **Prompt Template:** The structural requirements for folklore generation
- **Generated Story:** The actual LLM output

### Metric Computation
- Embeddings: OpenAI text-embedding-3-large
- Similarity: Cosine similarity between prompt and story embeddings
- Scale: 0.0 (no similarity) to 1.0 (perfect match)

### Statistical Tests
- One-way ANOVA (model effect on adherence)
- Tukey HSD post-hoc comparisons
- Effect size (η²)

---

## Results

### 1. Overall Adherence Statistics

| Statistic | Value |
|-----------|:-----:|
| Overall Mean | 0.358 |
| Overall SD | 0.076 |
| Minimum | 0.184 |
| Maximum | 0.636 |
| n | 599 |

**Interpretation:** Average adherence of 35.8% suggests models follow roughly one-third of the structural prompt—significant deviation is common.

### 2. Model Adherence Ranking

| Rank | Model | Mean | SD | Median | Min | Max |
|:----:|-------|:----:|:--:|:------:|:---:|:---:|
| 1 | **Qwen-3-8B** | **0.391** | 0.083 | 0.389 | 0.238 | 0.636 |
| 2 | GPT-5-mini | 0.382 | 0.075 | 0.379 | 0.249 | 0.610 |
| 3 | Gemini-3-Flash | 0.365 | 0.063 | 0.371 | 0.204 | 0.523 |
| 4 | Mistral-Large | 0.341 | 0.074 | 0.332 | 0.210 | 0.566 |
| 5 | **GPT-5.1** | **0.313** | 0.057 | 0.308 | 0.184 | 0.507 |

**Adherence Gap:** 0.078 (7.8% of scale between best and worst)

### 3. ANOVA Results

| Statistic | Value | Interpretation |
|-----------|:-----:|----------------|
| F(4, 594) | **23.94** | Highly significant |
| p-value | **< 0.001** | Significant |
| η² | **0.139** | Medium-large effect |

**Interpretation:** Model choice explains **13.9%** of variance in prompt adherence—a meaningful effect.

### 4. Tukey HSD Post-Hoc Comparisons

| Comparison | Mean Diff | p-adj | Significant? |
|------------|:---------:|:-----:|:------------:|
| GPT-5.1 vs Qwen | **-0.078** | < 0.001 | ✓ Yes |
| GPT-5.1 vs GPT-5-mini | -0.069 | < 0.001 | ✓ Yes |
| GPT-5.1 vs Gemini | -0.052 | < 0.001 | ✓ Yes |
| GPT-5.1 vs Mistral | -0.029 | 0.017 | ✓ Yes |
| Qwen vs Mistral | +0.050 | < 0.001 | ✓ Yes |
| Qwen vs Gemini | +0.026 | 0.036 | ✓ Yes |
| GPT-5-mini vs Mistral | +0.040 | < 0.001 | ✓ Yes |
| Qwen vs GPT-5-mini | +0.009 | 0.842 | ❌ No |
| Gemini vs Mistral | -0.024 | 0.077 | ❌ No |
| Gemini vs GPT-5-mini | +0.017 | 0.355 | ❌ No |

**Key Finding:** GPT-5.1 is significantly different from ALL other models (lowest adherence). Qwen and GPT-5-mini form a high-adherence cluster.

### 5. The GPT-5.1 Paradox

| Dimension | GPT-5.1 Rank | Implication |
|-----------|:------------:|-------------|
| Cultural Accuracy (RQ1) | #1 Best | Highest quality |
| Narrative Coherence (RQ3) | #1 Best | Best storytelling |
| **Prompt Adherence (RQ7)** | **#5 Worst** | **Lowest compliance** |

**Interpretation:** GPT-5.1 produces superior content by **ignoring** structural constraints—it prioritizes narrative quality over template obedience.

**Possible Explanations:**
1. **RLHF Training:** GPT-5.1 may be trained to produce engaging content, deprioritizing format compliance
2. **Creative Freedom:** Better models exercise more "artistic license"
3. **Prompt Interpretation:** GPT-5.1 may interpret prompts more abstractly

### 6. Story Type Adherence

| Rank | Story Type | Mean Adherence |
|:----:|------------|:--------------:|
| 1 | Human–Nature Relationship | **0.465** |
| 2 | Trickster or Clever Figure | 0.402 |
| 3 | Everyday Life Narrative | 0.380 |
| 4 | Change and Continuity | 0.370 |
| 5 | Community Crisis | 0.345 |
| 6 | Rite of Passage | 0.342 |
| 7 | Moral Transgression | 0.339 |
| 8 | Origin Story | 0.328 |
| 9 | Sacred or Forbidden Space | 0.321 |
| 10 | **Wisdom of an Elder** | **0.291** |

**Pattern:** Concrete, action-oriented story types (Human-Nature, Trickster) yield higher adherence than abstract types (Wisdom, Sacred Space).

### 7. Culture Adherence (Uniform)

| Culture | Mean | Rank |
|---------|:----:|:----:|
| Rakhine | 0.368 | 1 |
| Marma | 0.367 | 2 |
| Manipuri | 0.366 | 3 |
| ... | ... | ... |
| Khasi | 0.348 | 12 |

**Range:** 0.348 to 0.368 (spread = 0.020)

**Finding:** Culture has minimal effect on adherence—only 2% variation across 12 cultures.

### 8. Best and Worst Combinations

**Top Adherence (Best Compliance):**

| Model | Culture | Story Type | Adherence |
|-------|---------|------------|:---------:|
| Qwen-3-8B | Marma | Human–Nature | **0.636** |
| GPT-5-mini | Rakhine | Human–Nature | 0.610 |
| GPT-5-mini | Marma | Human–Nature | 0.592 |
| GPT-5-mini | Oraon | Human–Nature | 0.585 |
| Qwen-3-8B | Mro | Trickster | 0.581 |

**Bottom Adherence (Least Compliance):**

| Model | Culture | Story Type | Adherence |
|-------|---------|------------|:---------:|
| GPT-5.1 | Oraon | Wisdom of Elder | **0.184** |
| GPT-5.1 | Bengali | Wisdom of Elder | 0.197 |
| GPT-5.1 | Garo | Rite of Passage | 0.201 |
| Gemini | Santal | Origin Story | 0.204 |
| GPT-5.1 | Mro | Wisdom of Elder | 0.204 |

**Pattern:** GPT-5.1 + "Wisdom of an Elder" is a consistently low-adherence combination.

---

## Key Findings

### 1. Model Hierarchy (Adherence)

```
Qwen (0.39) ≈ GPT-5-mini (0.38) > Gemini (0.37) > Mistral (0.34) > GPT-5.1 (0.31)
```

This hierarchy is **inverse** to quality rankings from RQ1.

### 2. Quality-Adherence Trade-off

| Model | Quality Rank | Adherence Rank | Pattern |
|-------|:------------:|:--------------:|---------|
| GPT-5.1 | #1 | #5 | High quality, low adherence |
| Qwen | #5 | #1 | Low quality, high adherence |

**Correlation:** Quality and adherence appear **negatively correlated** at the model level.

### 3. Story Type Influences Adherence

Concrete narrative types (Human-Nature: 0.465) produce 60% higher adherence than abstract types (Wisdom of Elder: 0.291).

### 4. Culture Does Not Affect Adherence

All 12 cultures fall within a 2% adherence band—culture is irrelevant to structural compliance.

---

## Discussion

### Why Does GPT-5.1 Deviate from Prompts?

Several mechanisms may explain this counterintuitive finding:

1. **Instruction Hierarchy:** GPT-5.1 may prioritize "write a good story" over "follow this exact structure"
2. **Training Objectives:** RLHF optimizes for human preference, not template compliance
3. **Abstraction Capability:** More capable models interpret prompts as guidelines, not scripts
4. **Creative Emergence:** Quality emerges from structural deviation

### Is Low Adherence Bad?

**Not necessarily.** The data suggests:
- High adherence → Formulaic, lower-quality output
- Low adherence → Creative, higher-quality output

**However:** Applications requiring strict formatting (e.g., database entries, structured reports) would prefer high-adherence models.

### Implications for Prompt Engineering

| Goal | Recommended Model |
|------|-------------------|
| Maximum creativity | GPT-5.1 |
| Template compliance | Qwen-3-8B |
| Balance of both | GPT-5-mini |

---

## Practical Implications

### For Researchers
- **Adherence ≠ Quality:** Do not use adherence as a proxy for output quality
- **Measure both dimensions:** Include both metrics in evaluation frameworks
- **Model selection depends on goal:** Format-critical vs. quality-critical tasks

### For Practitioners
- **Strict formatting tasks:** Use Qwen-3-8B or GPT-5-mini
- **Creative generation:** Use GPT-5.1 (accept format variation)
- **Hybrid approach:** Generate with GPT-5.1, reformat with post-processing

### For Developers
- **Fine-tuning targets:** Adherence can likely be improved with targeted training
- **Prompt redesign:** More explicit structural cues may improve compliance
- **Output validation:** Implement structural validation for format-critical applications

---

## Limitations

1. **Adherence Definition:** Cosine similarity may not capture all structural elements
2. **Prompt Complexity:** Single prompt template—results may differ with varied prompts
3. **Correlation vs Causation:** Quality-adherence trade-off is observational
4. **Sample Size:** Single story per combination limits precision

---

## Conclusion

**Model selection significantly impacts prompt adherence** (η² = 0.139). Qwen-3-8B achieves the highest structural fidelity (M = 0.391), while GPT-5.1 shows the lowest (M = 0.313).

**The GPT-5.1 Paradox:** The best-performing model on quality metrics is the least obedient to structural prompts. This suggests a **trade-off between creativity and compliance**—superior models may deviate from templates to produce better content.

**Practical Recommendation:** Choose models based on task requirements:
- **Quality-critical:** GPT-5.1 (accept structural variation)
- **Format-critical:** Qwen-3-8B or GPT-5-mini (accept quality trade-off)

---

## Statistical Artifacts

| File | Description |
|------|-------------|
| `rq7_adherence_scores.csv` | Per-story adherence scores |
| `rq7_model_stats.csv` | Model-level statistics |
| `rq7_culture_stats.csv` | Culture-level statistics |
| `rq7_story_type_stats.csv` | Story type statistics |
| `rq7_anova_models.csv` | ANOVA results |
| `rq7_tukey_hsd.csv` | Post-hoc comparisons |
| `rq7_best_combinations.csv` | Highest adherence combos |
| `rq7_worst_combinations.csv` | Lowest adherence combos |
| `rq7_pivot_model_culture.csv` | Model × Culture pivot |
| `rq7_model_culture_stats.csv` | Detailed breakdown |
| `rq7_summary.csv` | Executive summary |
| `rq7_boxplot_adherence_by_model.png` | Model comparison |
| `rq7_violin_adherence_by_model.png` | Distribution visualization |
| `rq7_bar_adherence_ranking.png` | Ranking chart |
| `rq7_heatmap_model_culture.png` | Heatmap |
| `rq7_histogram_adherence.png` | Score distribution |

---

*Analysis conducted: January 2026*  
*Dataset: Bangladeshi Folklore Story Generation Evaluation (n=599)*
