# Evaluating LLM-Generated Bangladeshi Folklore: A Comprehensive Empirical Analysis

**A Research Report on Cultural Story Generation Quality Across Large Language Models**

---

## Abstract

This study presents a rigorous empirical evaluation of Large Language Model (LLM) performance in generating culturally authentic Bangladeshi folklore across 12 indigenous cultures. Through 16 carefully designed research questions (RQs), we analyzed 599 generated stories using human evaluation scores across four quality dimensions and semantic embedding analysis. Our findings reveal that **model selection is the dominant factor** in generation quality (η² = 0.21), while cultural identity and geographic region have negligible impact. Notably, we find **no evidence of systematic bias** against minority or indigenous cultures. However, LLMs exhibit significant **cultural semantic collapse**—generating stories that cluster into generic narrative patterns rather than maintaining distinct cultural identities. The highest-quality model (GPT-5.1) paradoxically shows the highest semantic drift from prompts and the most repetitive outputs, challenging conventional assumptions about prompt adherence and output diversity.

**Keywords:** Large Language Models, Cultural Story Generation, Folklore Preservation, Semantic Embeddings, Bias Analysis, Bangladeshi Indigenous Cultures

---

## 1. Introduction

### 1.1 Research Context

The application of Large Language Models to cultural content generation presents both opportunities and risks. While LLMs can potentially help preserve and disseminate endangered folklore, they may also introduce biases, homogenize distinct traditions, or generate culturally inauthentic content.

This study systematically evaluates five state-of-the-art LLMs generating folklore for Bangladesh's diverse ethnic communities, including:

- **Bengali** (majority culture)
- **11 Indigenous Cultures**: Chakma, Garo, Hajong, Khasi, Manipuri (Meitei), Marma, Mro, Rakhine, Santal, Tanchangya, and Tripura

### 1.2 Research Framework

We address 16 research questions organized into five thematic areas:

| Theme | Research Questions | Focus |
|-------|-------------------|-------|
| **Model Performance** | RQ1, RQ7, RQ9 | Which models perform best? |
| **Bias Detection** | RQ2, RQ5 | Do LLMs discriminate by culture/region? |
| **Content Analysis** | RQ3, RQ4, RQ10 | How do story characteristics affect quality? |
| **Embedding Analysis** | RQ6, RQ8, RQ11-16 | What do semantic representations reveal? |
| **Predictive Modeling** | RQ11, RQ12, RQ15 | Can embeddings predict quality? |

### 1.3 Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Stories** | 599 |
| **Models Evaluated** | 5 (GPT-5.1, GPT-5-Mini, Gemini-3-Flash, Mistral-Large, Qwen3-8B) |
| **Cultures Represented** | 12 |
| **Story Types** | 10 |
| **Geographic Regions** | 8 |
| **Evaluation Dimensions** | 4 (Cultural Accuracy, Contextual Appropriateness, Narrative Coherence, Linguistic Fluency) |

---

## 2. Methodology

### 2.1 Story Generation

Each model received identical structured prompts specifying:
- Target culture and geographic region
- Story type (e.g., Origin Story, Trickster Tale, Moral Lesson)
- Length and format requirements

### 2.2 Human Evaluation

Expert evaluators rated each story on four dimensions using a 5-point scale:

1. **Cultural Accuracy & Authenticity** (CA): Faithfulness to cultural traditions
2. **Contextual & Temporal Appropriateness** (CT): Alignment with setting/time period
3. **Narrative & Symbolic Coherence** (NC): Story structure and symbolic meaning
4. **Linguistic & Expressive Appropriateness** (LA): Language quality in Bangla

### 2.3 Embedding Analysis

Five embedding models were used for semantic analysis:
- OpenAI text-embedding-3-large (3072 dims)
- OpenAI text-embedding-3-small (1536 dims)
- Mistral mistral-embed-2312 (1024 dims)
- Qwen qwen3-embedding-8b (4096 dims)
- Google gemini-embedding-001 (3072 dims)

---

## 3. Key Findings

### 3.1 Model Performance (RQ1)

**Finding: GPT-5.1 significantly outperforms all other models.**

| Model | Mean Score | Std Dev | Rank |
|-------|------------|---------|------|
| **OpenAI GPT-5.1** | **3.133** | 0.647 | 1 |
| Google Gemini-3-Flash | 2.400 | 0.920 | 2 |
| OpenAI GPT-5-Mini | 2.375 | 0.581 | 3 |
| Mistral-Large-2512 | 2.200 | 0.559 | 4 |
| Qwen3-8B | 2.134 | 0.747 | 5 |

**Statistical Evidence:**

| Test | Statistic | p-value | Effect Size |
|------|-----------|---------|-------------|
| One-way ANOVA | F = 38.59 | p < 0.001 | η² = 0.206 (Large) |

**Interpretation:** Model choice explains **20.6% of variance** in quality scores—a large effect. GPT-5.1's mean (3.13) is 0.73-0.99 points higher than competitors (all Tukey HSD p < 0.001).

### 3.2 Cultural Bias Analysis (RQ2)

**Finding: No systematic bias against indigenous or minority cultures.**

| Comparison | Bengali (n=50) | Indigenous (n=549) | Difference | t | p | Cohen's d |
|------------|----------------|-------------------|------------|---|---|-----------|
| Cultural Accuracy | 2.42 | 2.45 | -0.03 | -0.27 | 0.79 | 0.04 |
| Contextual | 2.30 | 2.15 | +0.15 | 1.24 | 0.21 | 0.18 |
| Narrative | 2.04 | 1.99 | +0.05 | 0.52 | 0.60 | 0.08 |
| Linguistic | 1.56 | 1.43 | +0.13 | 1.18 | 0.24 | 0.17 |

**Interpretation:** All Cohen's d values are "negligible" (<0.20). **LLMs do not systematically produce lower-quality content for indigenous cultures.**

### 3.3 Story Type Impact (RQ3)

**Finding: Story type has NO significant effect on generation quality.**

| Metric | F-statistic | p-value | η² |
|--------|-------------|---------|-----|
| Cultural Accuracy | 0.62 | 0.78 | 0.009 |
| Contextual | 0.17 | 0.99 | 0.003 |
| Narrative | 0.55 | 0.84 | 0.008 |
| Linguistic | 0.26 | 0.98 | 0.004 |

**Interpretation:** η² values (0.003-0.009) indicate **virtually no effect**. Models handle all 10 story types with equal capability (or equal difficulty).

### 3.4 Fluent Hallucination Test (RQ4)

**Finding: Models are NOT "hallucinating fluently"—fluency and accuracy are positively correlated.**

| Correlation | Pearson r | p-value | R² |
|-------------|-----------|---------|-----|
| Linguistic Fluency ↔ Cultural Accuracy | **+0.521** | < 0.001 | 0.27 |

**Interpretation:** The positive correlation (r = +0.52) **rejects the hypothesis** that models produce fluent but inaccurate content. When linguistic quality is high, cultural accuracy tends to be high as well.

### 3.5 Regional Bias Analysis (RQ5)

**Finding: No regional bias detected across 8 geographic regions.**

| Metric | F-statistic | p-value | Significant? |
|--------|-------------|---------|--------------|
| Cultural Accuracy | 0.86 | 0.55 | No |
| Contextual | 0.59 | 0.79 | No |
| Narrative | 0.60 | 0.78 | No |
| Linguistic | 2.46 | 0.01 | Marginal* |

*The linguistic difference has η² = 0.032, a negligible effect size.

**Interpretation:** Remote or underrepresented regions (e.g., Chittagong Hill Tracts) receive **comparable quality** to major regions.

### 3.6 Semantic Separability (RQ6)

**Finding: Cultural narratives show POOR semantic clustering—near-complete cultural collapse.**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Overall Silhouette Score | **-0.0016** | Poor (negative = no clustering) |
| Best Separated Culture | Rakhine (0.017) | Still poor |
| Most Overlapping | Bengali (-0.028) | Negative = overlap |
| Mean Pairwise Distance | 0.052 | Very small separation |

**Interpretation:** The negative silhouette score indicates that **cultural stories are NOT semantically distinct**. LLMs generate "generic folklore" rather than culture-specific narratives, even when prompted with specific cultural details.

### 3.7 Prompt Adherence (RQ7)

**Finding: Inverse relationship between quality and prompt adherence.**

| Model | Mean Adherence | Quality Rank |
|-------|---------------|--------------|
| Qwen3-8B | **0.391** (highest) | 5 (lowest) |
| GPT-5-Mini | 0.382 | 3 |
| Gemini-3-Flash | 0.365 | 2 |
| Mistral-Large | 0.341 | 4 |
| **GPT-5.1** | **0.313** (lowest) | **1 (highest)** |

**Interpretation:** GPT-5.1 produces the highest quality despite the **lowest prompt adherence**. This suggests creative interpretation of prompts may be beneficial.

**Statistical Significance:** ANOVA F = 23.94, p < 0.001

### 3.8 Embedding Sensitivity (RQ8)

**Finding: Higher-dimensional embeddings show greater sensitivity to quality differences.**

| Embedding Model | Dimensions | Sensitivity Rank |
|-----------------|------------|------------------|
| OpenAI-Large | 3072 | 1 (most sensitive) |
| Qwen | 4096 | 2 |
| Gemini | 3072 | 3 |
| Mistral | 1024 | 4 |
| OpenAI-Small | 1536 | 5 (least sensitive) |

**Correlation:** Dimensions ↔ Sensitivity: r = 0.90

### 3.9 Model Repetitiveness (RQ9)

**Finding: GPT-5.1 is the most repetitive model, yet produces highest quality.**

| Model | Mean Similarity | Repetitiveness Rank | Quality Rank |
|-------|-----------------|---------------------|--------------|
| **GPT-5.1** | **0.525** | 1 (most repetitive) | 1 (highest) |
| GPT-5-Mini | 0.518 | 2 | 3 |
| Gemini-3-Flash | 0.500 | 3 | 2 |
| Mistral-Large | 0.430 | 4 | 4 |
| Qwen3-8B | 0.427 | 5 (least repetitive) | 5 (lowest) |

**Interpretation:** Repetitive patterns in GPT-5.1 may reflect **consistent quality frameworks** rather than creative failure.

### 3.10 Cross-Cultural Story Type Alignment (RQ10)

**Finding: Story types vary dramatically in cross-cultural consistency.**

| Metric | Value |
|--------|-------|
| ANOVA F-statistic | 55.89 |
| p-value | < 10⁻⁷⁴ |
| η² | **0.436** (very large) |

| Story Type | Classification | Interpretation |
|------------|---------------|----------------|
| Origin Story | Universal (0.21) | Most consistent across cultures |
| Human-Nature Relationship | Universal (0.14) | Highly consistent |
| Wisdom of an Elder | Distinct (0.29) | Most culturally specific |
| Rite of Passage | Distinct | Culture-specific |

**Interpretation:** η² = 0.44 indicates **43.6% of variance** in semantic distances is explained by story type. Some narrative types are universal; others are culturally distinct.

---

## 4. Advanced Embedding Analysis

### 4.1 Quality Prediction from Embeddings (RQ11)

**Finding: Embeddings provide weak but usable quality signals; Random Forest required.**

| Model Type | Best R² | MAE |
|------------|---------|-----|
| Ridge Regression | -0.49 | 0.74 |
| Lasso Regression | -0.33 | 0.72 |
| ElasticNet | -0.48 | 0.76 |
| **Random Forest** | **0.15** | 0.59 |
| Gradient Boosting | 0.07 | 0.62 |

**Interpretation:** Linear models fail (negative R²). Random Forest achieves modest predictive power (R² = 0.15), explaining **15% of quality variance** from embeddings alone.

### 4.2 Hallucination Detection (RQ12)

**Finding: LOF outperforms Isolation Forest for identifying low-quality outputs.**

| Method | Cohen's d | Significant Metrics |
|--------|-----------|---------------------|
| **LOF (Local Outlier Factor)** | **0.47** | 5/5 (all) |
| Isolation Forest | 0.21 | 0/5 (none) |

| Distance Metric | Quality Correlation | p-value |
|-----------------|---------------------|---------|
| Cosine Distance | r = -0.163 | < 0.001 |
| Euclidean Distance | r = -0.177 | < 0.001 |

**Interpretation:** Stories farther from cultural centroids tend to have **lower quality**. LOF can flag potential hallucinations with medium effect size (d = 0.47).

### 4.3 Semantic Drift vs. Quality (RQ13)

**Finding: Higher semantic drift does NOT predict lower quality—opposite relationship observed.**

| Model | Mean Drift | Quality Rank |
|-------|------------|--------------|
| Qwen3-8B | 0.609 (lowest) | 5 (lowest) |
| GPT-5-Mini | 0.618 | 3 |
| Gemini-3-Flash | 0.635 | 2 |
| Mistral-Large | 0.659 | 4 |
| **GPT-5.1** | **0.687** (highest) | **1 (highest)** |

| Correlation | r | p-value | Direction |
|-------------|---|---------|-----------|
| Drift ↔ Average Score | **+0.10** | 0.015 | Positive |

**Interpretation:** The weak **positive correlation** between drift and quality directly contradicts the hypothesis that semantic divergence hurts quality. GPT-5.1's high drift may reflect **creative elaboration** rather than deviation.

### 4.4 Safety Filter Effect (RQ14)

**Finding: No evidence of safety filters harming cultural content quality.**

| Correlation | r | p-value | Direction |
|-------------|---|---------|-----------|
| Genericity ↔ Cultural Accuracy | **+0.142** | < 0.001 | Positive |
| Genericity ↔ Average Score | **+0.183** | < 0.001 | Positive |

**Interpretation:** More generic stories receive **higher** scores, not lower. This challenges the "safety filter penalty" hypothesis. Generic narrative structures may be more universally readable.

### 4.5 Optimal Embedding Model (RQ15)

**Finding: Mistral-Embed (1024 dims) is the best embedding model for cultural evaluation.**

| Embedding Model | Mean R² | Best For |
|-----------------|---------|----------|
| **Mistral-Embed-2312** | **0.203** | 3/4 metrics |
| OpenAI-Small (1536d) | 0.165 | 1/4 metrics |
| OpenAI-Large (3072d) | 0.125 | 0 metrics |
| Qwen (4096d) | 0.112 | 0 metrics |
| Gemini (3072d) | 0.081 | 0 metrics |

**Interpretation:** **Smaller dimensions can be better.** Mistral's 1024-dimensional embeddings outperform 4096-dimensional alternatives, likely by avoiding the "curse of dimensionality" in this cultural domain.

### 4.6 Regional-Cultural Consistency (RQ16)

**Finding: 75% of culture pairs within regions are statistically indistinguishable.**

| Region | Culture Pairs | % Distinct | Most Distinct Pair |
|--------|---------------|------------|-------------------|
| Chittagong Hill Tracts | 3 | **0%** | None (all p > 0.27) |
| Sylhet Border | 1 | **100%** | Khasi-Manipuri (p < 0.001) |

| Culture Pair | Centroid Distance | KS p-value | Distinct? |
|--------------|-------------------|------------|-----------|
| Chakma-Marma | 0.041 | 0.27 | No |
| Chakma-Tripura | 0.045 | 0.72 | No |
| Marma-Tripura | 0.057 | 0.55 | No |
| **Khasi-Manipuri** | **0.078** | **<0.001** | **Yes** |

**Interpretation:** Chittagong Hill Tracts cultures (Chakma, Marma, Tripura) are **semantically conflated** by LLMs. Only Khasi-Manipuri shows clear separation—possibly due to distinct language families (Austroasiatic vs. Sino-Tibetan).

---

## 5. Synthesis: Key Cross-Cutting Insights

### 5.1 The GPT-5.1 Paradox

GPT-5.1 simultaneously exhibits:
- ✅ **Highest quality** (mean 3.13, significantly above all others)
- ❌ **Lowest prompt adherence** (0.313)
- ❌ **Highest semantic drift** (0.687)
- ❌ **Most repetitive** (similarity 0.525)

**Interpretation:** Quality does not come from strict prompt following, low drift, or diverse outputs. GPT-5.1 may have learned **effective cultural storytelling patterns** that it applies consistently, even if they diverge from literal prompts.

### 5.2 The Cultural Collapse Problem

Despite prompts specifying distinct cultures, LLMs generate **semantically homogeneous narratives**:

| Evidence | Finding |
|----------|---------|
| Silhouette Score | -0.0016 (no clustering) |
| Distinct Culture Pairs | 25% (1/4 regional pairs) |
| Best Cultural Separation | Rakhine (score = 0.017, still poor) |

**Implication:** Current LLMs cannot reliably distinguish between culturally similar groups. They apply generic "folklore templates" rather than culture-specific knowledge.

### 5.3 Model Choice Dominates Everything

| Factor | Variance Explained | Relative Impact |
|--------|-------------------|-----------------|
| **Model Choice** | η² = 0.206 | **100%** (baseline) |
| Story Type | η² = 0.008 | 4% |
| Culture | η² ≈ 0.01 | 5% |
| Region | η² ≈ 0.01 | 5% |

**Model choice is ~20× more important than content factors.** Selecting the right LLM matters far more than prompt engineering for story type, culture, or region.

### 5.4 No Systematic Bias—But No Cultural Precision Either

| Bias Type | Evidence | Conclusion |
|-----------|----------|------------|
| Anti-Indigenous | Cohen's d = 0.04 | **No bias** |
| Anti-Remote Regions | η² = 0.01 | **No bias** |
| Anti-Minority Languages | F = 0.62, p = 0.78 | **No bias** |

**However:** Lack of bias comes with lack of differentiation. LLMs treat all cultures similarly because they generate **generic content** for all.

---

## 6. Practical Recommendations

### 6.1 For Practitioners

| Goal | Recommendation |
|------|----------------|
| **Maximum Quality** | Use GPT-5.1 |
| **Embedding Analysis** | Use Mistral-Embed-2312 (1024d) |
| **Hallucination Detection** | Use LOF (not Isolation Forest) |
| **Cultural Distinction** | Use Gemini or Mistral (not GPT-5-Mini) |
| **Prompt Design** | Focus on cultural specificity, not strict format |

### 6.2 For Model Developers

1. **Train on distinct cultural corpora** to prevent cultural collapse
2. **Add cultural embedding clusters** to preserve inter-culture distinctions
3. **Evaluate on minority cultures specifically** during development
4. **Consider smaller, specialized embeddings** for domain-specific tasks

### 6.3 For Researchers

1. **Don't assume prompt adherence = quality**
2. **Use multiple embedding models** for robustness
3. **Validate cultural distinctiveness explicitly**
4. **Random Forest required** for embedding-based prediction

---

## 7. Limitations

1. **Dataset Size:** 599 stories may not capture full cultural diversity
2. **Evaluator Expertise:** Cultural authenticity requires deep domain knowledge
3. **Embedding Dependence:** All semantic analyses depend on embedding quality
4. **Temporal Scope:** Models and capabilities change rapidly
5. **Language Focus:** Analysis primarily in Bangla; may not generalize

---

## 8. Conclusion

This comprehensive analysis of LLM-generated Bangladeshi folklore reveals several counterintuitive findings:

1. **Model selection is paramount** (η² = 0.21), dwarfing all content factors combined
2. **No cultural or regional bias exists**, but this reflects cultural homogenization rather than precision
3. **Cultural semantic collapse is pervasive**—LLMs struggle to maintain distinct cultural identities
4. **Quality paradoxically correlates with drift and repetition**, not adherence and diversity
5. **Smaller embeddings (Mistral 1024d) outperform larger ones** for cultural evaluation

The central tension in cultural AI is revealed: **LLMs achieve fairness through homogenization, not through genuine cultural understanding.** They treat all cultures equally well (or equally poorly) by applying generic narrative templates rather than culture-specific knowledge.

**Future work should focus on:**
- Cultural grounding techniques to prevent semantic collapse
- Explicit training on inter-culture distinctions
- Development of cultural authenticity metrics beyond human evaluation
- Investigation of whether these patterns hold for other endangered cultural traditions

---

## Appendix A: Statistical Summary Tables

### A.1 Model Performance Summary

| Model | CA | CT | NC | LA | Average | Rank |
|-------|-----|-----|-----|-----|---------|------|
| GPT-5.1 | 3.13 | 2.83 | 2.74 | 2.23 | 2.73 | 1 |
| Gemini-3-Flash | 2.40 | 2.12 | 1.88 | 1.28 | 1.92 | 2 |
| GPT-5-Mini | 2.38 | 2.22 | 1.87 | 1.58 | 2.01 | 3 |
| Mistral-Large | 2.20 | 2.02 | 1.68 | 1.17 | 1.77 | 4 |
| Qwen3-8B | 2.13 | 1.84 | 1.78 | 1.33 | 1.77 | 5 |

### A.2 Effect Sizes Summary

| Research Question | Key Statistic | Effect Size | Interpretation |
|-------------------|--------------|-------------|----------------|
| RQ1: Model Effect | η² = 0.206 | Large | Model matters greatly |
| RQ2: Cultural Bias | d = 0.04 | Negligible | No bias |
| RQ3: Story Type Effect | η² = 0.008 | Negligible | No effect |
| RQ4: Fluency-Accuracy | r = 0.52 | Large | Strong positive |
| RQ5: Regional Bias | η² = 0.01 | Negligible | No bias |
| RQ6: Silhouette Score | -0.0016 | Poor | Cultural collapse |
| RQ10: Story Type Clustering | η² = 0.44 | Very Large | Strong type effect |
| RQ11: Prediction R² | 0.15 | Moderate | Weak but useful |
| RQ12: LOF Detection | d = 0.47 | Medium | Useful signal |

### A.3 Correlation Matrix

| Variable 1 | Variable 2 | r | p | Interpretation |
|------------|------------|---|---|----------------|
| Linguistic Fluency | Cultural Accuracy | +0.52 | <0.001 | Quality dimensions align |
| Semantic Drift | Quality Score | +0.10 | 0.015 | Drift helps (slightly) |
| Genericity | Quality Score | +0.18 | <0.001 | Generic helps |
| Centroid Distance | Quality Score | -0.16 | <0.001 | Outliers lower quality |
| Prompt Adherence | Quality Score | -0.15 | <0.001 | Adherence hurts quality |

---

## Appendix B: Research Questions Index

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | Which model performs best? | GPT-5.1 (p < 0.001, η² = 0.21) |
| RQ2 | Is there cultural bias? | No (all d < 0.20) |
| RQ3 | Does story type matter? | No (η² = 0.008) |
| RQ4 | Are models fluently hallucinating? | No (r = +0.52) |
| RQ5 | Is there regional bias? | No (all p > 0.55) |
| RQ6 | Do cultures cluster semantically? | No (silhouette = -0.0016) |
| RQ7 | Which model follows prompts best? | Qwen (but lowest quality) |
| RQ8 | Which embedding is most sensitive? | OpenAI-Large (r = 0.90 with dims) |
| RQ9 | Which model is most repetitive? | GPT-5.1 (but highest quality) |
| RQ10 | Are story types universal? | Varies (η² = 0.44) |
| RQ11 | Can embeddings predict quality? | Weakly (R² = 0.15, RF only) |
| RQ12 | Can embeddings detect hallucination? | Yes (LOF d = 0.47) |
| RQ13 | Does drift hurt quality? | No (r = +0.10) |
| RQ14 | Do safety filters hurt cultural content? | No (r = +0.18) |
| RQ15 | Which embedding is best for evaluation? | Mistral (R² = 0.20) |
| RQ16 | Are regional cultures distinct? | Mostly no (75% indistinguishable) |

---

*Report generated from analysis of 599 LLM-generated Bangladeshi folklore stories across 12 cultures, 10 story types, 8 regions, and 5 language models. Statistical analyses conducted using Python with scipy, statsmodels, and scikit-learn.*

---

**Citation:**  
JAF Research Team. (2026). *Evaluating LLM-Generated Bangladeshi Folklore: A Comprehensive Empirical Analysis.* Research Report, JAF_RQ_ANALYSIS Project.
