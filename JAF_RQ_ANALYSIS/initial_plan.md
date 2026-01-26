Based on the dataset schema and analysis reports, here are **15+ Research Questions (RQs)** tailored to this specific dataset.

I have categorized them into **Metric-Based (CSV Analysis)**, **Vector-Space (Parquet Analysis)**, and **Hybrid (Linking both)** to maximize the potential of your data.

---

### **Part 1: Metric-Based RQs (Target: `evaluated_stories.csv`)**

*Focus: Statistical analysis of human evaluation scores against metadata.*

**RQ1: Comparative Model Efficacy in Cultural Authenticity**

* **Question:** Which LLM architecture (`google_gemini-3-flash`, `mistral-large`, `gpt-5`, etc.) yields the highest human-rated performance for preservation of indigenous cultural norms?
* **What to do:** Group by `model` and calculate the mean and variance of the `Cultural Accuracy & Authenticity` column. Perform an ANOVA test to determine if the differences between models are statistically significant.

**RQ2: Bias in Representation of Low-Resource Cultures**

* **Question:** Is there a significant performance disparity in generated story quality between dominant cultures (e.g., Bengali) and marginalized indigenous groups (e.g., Mro, Khasi)?
* **What to do:** Segment data by `culture`. Compare the average `Cultural Accuracy` and `Linguistic Appropriateness` scores of "Bengali" vs. the aggregated average of the 11 indigenous tribal groups.

**RQ3: Narrative Complexity vs. Model Performance**

* **Question:** Does the type of folklore narrative (e.g., "Origin Story" vs. "Everyday Life") impact the model's ability to maintain narrative coherence?
* **What to do:** Pivot the data using `story_type` as rows and evaluation metrics as values. Identify which story types consistently receive the lowest `Narrative & Symbolic Coherence` scores across all models.

**RQ4: The Correlation between Linguistic Fluency and Cultural Accuracy**

* **Question:** Does high linguistic proficiency in standard Bangla imply a high understanding of cultural nuance, or are models "hallucinating fluently"?
* **What to do:** Calculate the Pearson correlation coefficient between `Linguistic & Expressive Appropriateness (Bangla)` and `Cultural Accuracy & Authenticity`. A low correlation suggests models write good Bangla but fail at cultural facts.

**RQ5: Regional Contextual Performance**

* **Question:** Do models perform better on prompts set in geographically prominent regions (e.g., "Mymensingh") compared to remote areas (e.g., "Chittagong Hill Tracts")?
* **What to do:** Join `evaluated_stories.csv` with the `region` column from the parquet metadata. Compare evaluation scores grouped by `region` to check for geographical training data bias.

---

### **Part 2: Vector-Space RQs (Target: `*.parquet` files)**

*Focus: Semantic analysis using embeddings from different models.*

**RQ6: Semantic Separability of Cultural Narratives**

* **Question:** Do high-dimensional embeddings distinctively cluster different cultures, or do they collapse distinct indigenous groups into a single "generic tribal" cluster?
* **What to do:** Apply t-SNE or UMAP dimensionality reduction on `story_embedding` columns from the parquet files. Color the points by `culture` and visually/mathematically inspect the separation (Silhouette Score) between groups like *Santal* vs. *Chakma*.

**RQ7: Prompt Adherence Quantification**

* **Question:** Which generation model stays truest to the semantic intent of the prompt scenario?
* **What to do:** For each row in the parquet files, calculate the **Cosine Similarity** between `scenario_embedding` (the input intent) and `story_embedding` (the output). Compare these similarity scores across the 5 generation `model`s.

**RQ8: Embedding Model Sensitivity Comparison**

* **Question:** Does a larger embedding dimension (e.g., `qwen3` at 4096d) capture more subtle narrative variations than smaller models (e.g., `mistral` at 1024d)?
* **What to do:** Compute the pairwise distances between stories of the *same* story type but *different* cultures across all 5 embedding files. The embedding model with the highest variance in these distances likely captures more nuance.

**RQ9: Detection of "Model Collapse" or Repetition**

* **Question:** Do certain models produce stories that are semantically repetitive across different prompts?
* **What to do:** Calculate the average cosine similarity of all generated `story_embedding` vectors *within* a single model group. A very high average similarity indicates the model is outputting generic/repetitive "safe" stories regardless of the specific prompt.

**RQ10: Cross-Cultural Semantic Alignment**

* **Question:** Are "Origin Stories" semantically similar across different cultures in the vector space (suggesting universal tropes), or are they distinct?
* **What to do:** Filter for `story_type = "Origin Story"`. Compute the centroid of embeddings for each `culture`. Analyze the Euclidean distance between these centroids to measure cultural distinctiveness in folklore structure.

---

### **Part 3: Hybrid RQs (Linking CSV Evaluations + Parquet Embeddings)**

*Focus: Using embeddings to explain or predict human scores.*

**RQ11: Automated Reward Modeling / Score Prediction**

* **Question:** Can high-dimensional story embeddings accurately predict human evaluation scores, effectively acting as an "AI Judge"?
* **What to do:** Train a Ridge Regression or Random Forest model using `story_embedding` (Parquet) as features (X) and `Cultural Accuracy` (CSV) as the target (Y). Report the R-squared value to determine predictive feasibility.

**RQ12: Identifying "Hallucination" in Vector Space**

* **Question:** Do stories with low "Contextual Appropriateness" scores manifest as outliers in the embedding space?
* **What to do:** Identify the centroid of stories with a score of 5 (perfect). Calculate the distance of stories with scores of 1 or 2 from this "ideal centroid." Test if low-quality stories lie significantly further away (outlier detection).

**RQ13: Semantic Drift vs. Human Perception**

* **Question:** Does a divergence between the Prompt Embedding and Story Embedding (semantic drift) correlate with lower human ratings for Coherence?
* **What to do:** Correlate the "Prompt-Story Cosine Similarity" (calculated from Parquet) with the `Narrative & Symbolic Coherence` score (from CSV). This validates if vector drift is a good proxy for human-perceived coherence.

**RQ14: The "Safety Filter" Effect**

* **Question:** Do models with high "refusal" rates or generic outputs (detected via low embedding variance) receive lower Cultural Authenticity scores?
* **What to do:** Identify stories with embeddings that are highly similar to a "generic safety response" centroid. Check their corresponding `Cultural Accuracy` scores in the CSV to see if "safe" outputs are penalized by human raters.

**RQ15: Dimensionality vs. Evaluation Correlation**

* **Question:** Which embedding model (`openai-large` vs. `gemini` vs. `qwen`) correlates best with human judgment, making it the best candidate for future automated evaluation?
* **What to do:** Repeat RQ11 (Regression) for *all 5 parquet files*. Compare which embedding source yields the highest accuracy in predicting the human scores.

**RQ16 (Bonus): Temporal/Regional Consistency**

* **Question:** Do stories generated for the same Region but different Cultures (e.g., Chittagong Hill Tracts -> Chakma vs. Marma) show appropriate semantic distinctiveness?
* **What to do:** Filter embeddings by `Region = "Chittagong Hill Tracts"`. Measure the overlap of distributions between `culture="Chakma"` and `culture="Marma"`. If they overlap completely, the model might be conflating distinct regional tribes.