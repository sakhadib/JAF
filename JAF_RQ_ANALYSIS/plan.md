# Research Question Plan: Bangladeshi Folklore Story Generation & Evaluation

## Data Availability Assessment

### Files Available
| File | Key Columns | Notes |
|------|-------------|-------|
| `evaluated_stories.csv` | story_id, **culture**, **model**, story, 4 evaluation metrics | Human ratings only |
| `parquet/*.parquet` (5 files) | **model**, **culture**, story, story_type, region, scenario, prompt, scenario_embedding, story_embedding | Full metadata + embeddings |

### Critical Finding: Data Linking Required
- **CSV lacks**: `story_type`, `region`, `scenario`, `prompt_id` — these exist ONLY in parquet files
- **Link Key**: `(model, culture, story)` or simply match on story text hash
- **Recommendation**: Create a unified dataframe by joining CSV evaluations with parquet metadata

---

## Feasibility Evaluation of Initial RQs

| RQ | Feasible? | Data Source | Join Required? |
|----|-----------|-------------|----------------|
| RQ1 | ✅ Yes | CSV only | No |
| RQ2 | ✅ Yes | CSV only | No |
| RQ3 | ⚠️ Yes | CSV + Parquet | **Yes** (story_type in parquet) |
| RQ4 | ✅ Yes | CSV only | No |
| RQ5 | ⚠️ Yes | CSV + Parquet | **Yes** (region in parquet) |
| RQ6 | ✅ Yes | Parquet only | No |
| RQ7 | ✅ Yes | Parquet only | No |
| RQ8 | ✅ Yes | Multiple Parquets | No |
| RQ9 | ✅ Yes | Parquet only | No |
| RQ10 | ✅ Yes | Parquet only | No |
| RQ11 | ✅ Yes | CSV + Parquet | **Yes** |
| RQ12 | ✅ Yes | CSV + Parquet | **Yes** |
| RQ13 | ✅ Yes | CSV + Parquet | **Yes** |
| RQ14 | ✅ Yes | CSV + Parquet | **Yes** |
| RQ15 | ✅ Yes | CSV + Parquet | **Yes** |
| RQ16 | ✅ Yes | Parquet only | No |

**Verdict**: All 16 RQs are answerable. 8 require data joining.

---

## Detailed Research Question Plan

### Phase 0: Data Preparation (PREREQUISITE)

**Task 0.1: Create Unified Dataset**
```
Action: Join evaluated_stories.csv with one parquet file on (model, culture, story)
Output: unified_dataset.parquet with all columns + evaluation scores
Columns: story_id, culture, model, story, story_type, region, scenario, prompt,
         Cultural_Accuracy, Contextual_Appropriateness, Narrative_Coherence, 
         Linguistic_Appropriateness, scenario_embedding, story_embedding
```

**Task 0.2: Validate Join Integrity**
```
Check: All 599 rows match between CSV and parquet
Check: No duplicate or missing stories after join
```

---

### Phase 1: Metric-Based Analysis (CSV-focused)

#### RQ1: Comparative Model Efficacy in Cultural Authenticity
**Question**: Which LLM yields the highest human-rated cultural authenticity?

| Step | Action | Method |
|------|--------|--------|
| 1.1 | Group by `model` | `df.groupby('model')` |
| 1.2 | Calculate descriptive stats | Mean, Std, Median for `Cultural Accuracy & Authenticity` |
| 1.3 | Statistical test | One-way ANOVA + Tukey HSD post-hoc |
| 1.4 | Visualize | Box plot / Violin plot by model |

**Expected Output**: Ranking of 5 models with significance levels

---

#### RQ2: Bias in Representation of Low-Resource Cultures
**Question**: Performance disparity between Bengali (dominant) vs. indigenous groups?

| Step | Action | Method |
|------|--------|--------|
| 2.1 | Create binary grouping | Bengali vs. Others (11 cultures aggregated) |
| 2.2 | Compare means | Independent samples t-test |
| 2.3 | Per-culture analysis | Group by all 12 cultures, rank by each metric |
| 2.4 | Visualize | Heatmap: culture × metric |

**Expected Output**: Statistical evidence of bias (or lack thereof)

---

#### RQ3: Narrative Complexity vs. Model Performance ⚠️ *Requires Join*
**Question**: Does story_type impact narrative coherence scores?

| Step | Action | Method |
|------|--------|--------|
| 3.1 | **Join** parquet to get `story_type` | Merge on (model, culture, story) |
| 3.2 | Pivot table | Rows: story_type, Cols: models, Values: mean(Narrative Coherence) |
| 3.3 | Statistical test | Two-way ANOVA (story_type × model) |
| 3.4 | Identify difficult types | Which story_types have lowest scores? |

**Expected Output**: Difficulty ranking of 10 story types

---

#### RQ4: Linguistic Fluency vs. Cultural Accuracy Correlation
**Question**: Are models "hallucinating fluently"?

| Step | Action | Method |
|------|--------|--------|
| 4.1 | Extract columns | `Linguistic_Appropriateness`, `Cultural_Accuracy` |
| 4.2 | Correlation analysis | Pearson + Spearman correlation coefficients |
| 4.3 | Scatter plot | With regression line and confidence interval |
| 4.4 | Per-model correlation | Check if correlation varies by model |

**Expected Output**: Correlation coefficient + interpretation

**Interpretation Guide**:
- r > 0.7: Strong link (fluency implies accuracy)
- r < 0.3: Weak link (models hallucinate fluently)

---

#### RQ5: Regional Contextual Performance ⚠️ *Requires Join*
**Question**: Better performance on prominent regions vs. remote areas?

| Step | Action | Method |
|------|--------|--------|
| 5.1 | **Join** to get `region` | 9 unique regions in parquet |
| 5.2 | Group by region | Calculate mean scores per region |
| 5.3 | Categorize regions | Prominent (Dhaka-adjacent) vs. Remote (Hill Tracts) |
| 5.4 | Statistical test | ANOVA across regions |

**Expected Output**: Regional performance ranking

---

### Phase 2: Vector-Space Analysis (Parquet-focused)

#### RQ6: Semantic Separability of Cultural Narratives
**Question**: Do embeddings cluster by culture or collapse into generic groups?

| Step | Action | Method |
|------|--------|--------|
| 6.1 | Load embeddings | Use `story_embedding` from one parquet (e.g., openai-large) |
| 6.2 | Dimensionality reduction | UMAP or t-SNE to 2D |
| 6.3 | Cluster quality | Silhouette Score by culture labels |
| 6.4 | Visualize | Scatter plot colored by culture |
| 6.5 | Pairwise analysis | Which cultures overlap most? |

**Expected Output**: Cluster visualization + separability metrics

---

#### RQ7: Prompt Adherence Quantification
**Question**: Which generation model stays truest to the prompt?

| Step | Action | Method |
|------|--------|--------|
| 7.1 | Compute similarity | `cosine_similarity(scenario_embedding, story_embedding)` per row |
| 7.2 | Group by `model` | Average similarity per generation model |
| 7.3 | Statistical test | ANOVA on similarity scores |
| 7.4 | Visualize | Box plot of adherence by model |

**Expected Output**: Model ranking by prompt adherence

---

#### RQ8: Embedding Model Sensitivity Comparison
**Question**: Does higher dimensionality capture more nuance?

| Step | Action | Method |
|------|--------|--------|
| 8.1 | For each of 5 embedding files | Load story_embedding |
| 8.2 | Same-type, different-culture distances | Filter by story_type, compute inter-culture distances |
| 8.3 | Measure variance | Which embedding model shows highest distance variance? |
| 8.4 | Compare dimensions | 1024 (mistral) vs 3072 (openai-large) vs 4096 (qwen) |

**Expected Output**: Ranking of embedding models by sensitivity

---

#### RQ9: Detection of Model Collapse / Repetition
**Question**: Do certain models produce semantically repetitive outputs?

| Step | Action | Method |
|------|--------|--------|
| 9.1 | Group by `model` | Get all story_embeddings per generation model |
| 9.2 | Intra-model similarity | Average pairwise cosine similarity within each model |
| 9.3 | Compare | High similarity = repetitive/generic outputs |
| 9.4 | Outlier detection | Which model has tightest cluster (most repetitive)? |

**Expected Output**: Repetitiveness score per model

---

#### RQ10: Cross-Cultural Semantic Alignment in Story Types
**Question**: Are "Origin Stories" universal or culturally distinct?

| Step | Action | Method |
|------|--------|--------|
| 10.1 | Filter | `story_type == "Origin Story"` |
| 10.2 | Compute centroids | Mean embedding per culture |
| 10.3 | Distance matrix | Euclidean distance between culture centroids |
| 10.4 | Repeat | For other story_types (Community Crisis, etc.) |
| 10.5 | Compare | Which story_type has most/least cultural variation? |

**Expected Output**: Cultural distinctiveness matrix per story type

---

### Phase 3: Hybrid Analysis (CSV + Parquet)

#### RQ11: Automated Score Prediction ("AI Judge")
**Question**: Can embeddings predict human evaluation scores?

| Step | Action | Method |
|------|--------|--------|
| 11.1 | **Join** data | Embeddings (X) + Evaluation scores (Y) |
| 11.2 | Train-test split | 80/20 stratified by culture |
| 11.3 | Model training | Ridge Regression / Random Forest |
| 11.4 | Evaluate | R², MAE, RMSE for each of 4 metrics |
| 11.5 | Feature importance | Which embedding dimensions matter? |

**Expected Output**: Predictive model performance metrics

---

#### RQ12: Identifying Hallucination in Vector Space
**Question**: Are low-quality stories outliers in embedding space?

| Step | Action | Method |
|------|--------|--------|
| 12.1 | Define "ideal" centroid | Mean embedding of stories with score=5 (if any) or score≥4 |
| 12.2 | Compute distances | Distance of each story from ideal centroid |
| 12.3 | Correlate with scores | Do low-score stories have high distances? |
| 12.4 | Outlier detection | Isolation Forest or LOF on embeddings |

**Expected Output**: Correlation between outlier status and quality scores

---

#### RQ13: Semantic Drift vs. Human Perception
**Question**: Does prompt-story divergence predict coherence scores?

| Step | Action | Method |
|------|--------|--------|
| 13.1 | Compute drift | `1 - cosine_similarity(scenario_emb, story_emb)` |
| 13.2 | Correlate | Pearson correlation with `Narrative & Symbolic Coherence` |
| 13.3 | Visualize | Scatter plot: drift vs. coherence score |
| 13.4 | Threshold analysis | What drift threshold indicates quality drop? |

**Expected Output**: Drift-coherence correlation coefficient

---

#### RQ14: The "Safety Filter" Effect
**Question**: Do generic/safe outputs receive lower cultural scores?

| Step | Action | Method |
|------|--------|--------|
| 14.1 | Compute "genericity" | Similarity to overall embedding centroid |
| 14.2 | Identify generic stories | Top 10% most similar to centroid |
| 14.3 | Compare scores | Generic vs. non-generic story scores |
| 14.4 | Per-model analysis | Which model produces most generic outputs? |

**Expected Output**: Genericity penalty quantification

---

#### RQ15: Best Embedding Model for Automated Evaluation
**Question**: Which embedding correlates best with human judgment?

| Step | Action | Method |
|------|--------|--------|
| 15.1 | For each of 5 parquet files | Repeat RQ11 (regression) |
| 15.2 | Compare R² scores | Across all 5 embedding models |
| 15.3 | Per-metric analysis | Which embedding best predicts which metric? |
| 15.4 | Recommend | Best embedding model for future auto-evaluation |

**Expected Output**: Embedding model ranking for evaluation tasks

---

#### RQ16: Regional-Cultural Consistency
**Question**: Are cultures in same region appropriately distinct?

| Step | Action | Method |
|------|--------|--------|
| 16.1 | Filter | `region == "Chittagong Hill Tracts"` (contains Chakma, Marma, Tripura, Mro) |
| 16.2 | Compute overlap | Distribution overlap between culture embeddings |
| 16.3 | Statistical test | KL Divergence or MMD between distributions |
| 16.4 | Repeat | For Sylhet region (Khasi, Manipuri) |

**Expected Output**: Intra-region cultural distinctiveness scores

---

## Implementation Priority

### Tier 1: Quick Wins (CSV-only, no join required)
1. **RQ1** - Model comparison (1-2 hours)
2. **RQ2** - Cultural bias analysis (1-2 hours)
3. **RQ4** - Correlation analysis (1 hour)

### Tier 2: Embedding Analysis (Parquet-only)
4. **RQ6** - Cultural clustering visualization (2-3 hours)
5. **RQ7** - Prompt adherence (1-2 hours)
6. **RQ9** - Model repetitiveness (1-2 hours)

### Tier 3: Hybrid Analysis (Requires data merge)
7. **Task 0.1** - Create unified dataset (30 min) ← **DO THIS FIRST**
8. **RQ3** - Story type analysis (2 hours)
9. **RQ11** - Score prediction model (3-4 hours)
10. **RQ13** - Semantic drift analysis (2 hours)

### Tier 4: Advanced Analysis
11. **RQ8** - Embedding comparison across 5 models (3 hours)
12. **RQ10** - Cross-cultural alignment (3 hours)
13. **RQ15** - Best embedding for evaluation (4 hours)

---

## Required Libraries

```python
# Core
pandas, numpy, pyarrow

# Statistics
scipy.stats (ttest_ind, f_oneway, pearsonr, spearmanr)
statsmodels (tukey_hsd, anova)

# ML / Embeddings
scikit-learn (cosine_similarity, Ridge, RandomForest, train_test_split)
umap-learn, sklearn.manifold (TSNE)
sklearn.metrics (silhouette_score, r2_score)

# Visualization
matplotlib, seaborn, plotly
```

---

## Output Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| `unified_dataset.parquet` | Parquet | Merged data with all columns |
| `rq_results.json` | JSON | Statistical test results for each RQ |
| `figures/` | PNG/SVG | Visualizations for each RQ |
| `models/` | Pickle | Trained prediction models (RQ11, RQ15) |
| `final_report.md` | Markdown | Summary of findings |

---

## Notes & Caveats

1. **Sample Size**: 599 stories may limit statistical power for fine-grained analysis (e.g., model × culture × story_type combinations)

2. **Score Distribution**: Evaluation scores are skewed (most scores are 1-3, very few 4-5). Consider:
   - Treating as ordinal, not continuous
   - Using non-parametric tests

3. **Embedding Model Selection**: For hybrid RQs, start with `openai_text-embedding-3-large` (3072d) as the primary embedding source

4. **Culture Imbalance**: Oraon (Kurukh) has 49 samples vs. 50 for others - minor but note in analysis
