### **Theme I: The Politics of Representation (Mainstream vs. Indigenous)**

*Focus: Does the AI function as a "colonial" tool that subsumes indigenous narratives into the hegemonic Bengali/Western worldview?*

**RQ1:** **The "Bengali Hegemony" Hypothesis**

* **Question:** To what extent do LLMs impose mainstream Bengali cultural signifiers (e.g., riverine metaphors, Islamic/Hindu customs) onto indigenous narratives (e.g., Hill Tracts cultures like Chakma or Marma) where they are contextually inappropriate?  
* **Data Source:** Compare Cultural Accuracy scores between "Bengali" rows and "Indigenous" rows (e.g., Santal, Garo).

**RQ2:** **Hallucination of Rituals in Low-Resource Cultures**

* **Question:** Does the frequency of "hallucinated" cultural artifacts (e.g., inventing non-existent rituals or deities) increase inversely with the digital presence of the culture (e.g., are Mro or Khumi stories more prone to fabrication than Bengali ones)?  
* **Data Source:** Correlation between Culture type and Narrative & Symbolic Coherence scores.

**RQ3:** **Linguistic Homogenization vs. Dialectal Nuance**

* **Question:** Since the prompts are in Standard Bangla, does the AI strip away the linguistic markers and distinct "voice" of indigenous oral traditions, rendering a "Santal" story indistinguishable in tone from a "Bengali" one?  
* **Data Source:** NLP analysis of story text; Linguistic & Expressive Appropriateness comparison across the 12 cultures.

**RQ4:** **Spatial "Flattening" in Embeddings**

* **Question:** Do the vector embeddings fail to distinguish between geographically distinct cultures (e.g., distinguishing "Plains" cultures like Santal from "Hill Tracts" cultures like Marma), clustering them all into a generic "Non-Western" blob?  
* **Data Source:** Clustering analysis (t-SNE/PCA) of the Parquet files labeled by Region from cultures.csv.

---

### **Theme II: Narrative Morphology & Genre (The Scenarios)**

*Focus: Analyzing how AI handles specific structural forms defined in your scenarios.csv.*

**RQ5:** **Cosmological Accuracy in "Origin Stories"**

* **Question:** In the "Origin Story" scenario, does the AI respect the distinct cosmogonies of each tribe (e.g., Santal creation myths involving water/earthworms) or does it revert to generic "Adam/Eve" or "Big Bang" tropes?  
* **Data Source:** Qualitative/Keyword analysis of Story\_Type="Origin Story" rows.

**RQ6:** **The "Wisdom of the Elder" Stereotype**

* **Question:** For the "Wisdom of an Elder" scenario, does the AI rely on the "Magical Negro" or "Noble Savage" trope for indigenous elders, portraying them as mystical rather than practical community leaders?  
* **Data Source:** Contextual & Temporal Appropriateness scores specifically for this scenario.

**RQ7:** **Collective vs. Individual Resolution in "Community Crisis"**

* **Question:** In "Community Crisis" scenarios, does the AI favor Western-style individual heroism (a single hero saves the village) over the communal/collective problem-solving typical of South Asian agrarian societies?  
* **Data Source:** Narrative analysis of resolution patterns in Story\_Type="Community Crisis".

**RQ8:** **Animism vs. Resource Management**

* **Question:** In "Human–Nature Relationship" scenarios, does the AI portray nature as an animate, spiritual partner (animism) or merely as a resource to be managed/protected (modern environmentalism)?  
* **Data Source:** Keyword frequency of spiritual vs. utilitarian terms in Story\_Type="Human-Nature Relationship".

---

### **Theme III: Computational Folkloristics (Vectors & Models)**

*Focus: Using the 5 embedding models to "measure" culture.*

**RQ9:** **The Latent Space of "Sacredness"**

* **Question:** Can vector embeddings distinguish between "Sacred" narratives (Origin, Moral Transgression) and "Secular" narratives (Community Crisis)? Do these genres form distinct clusters?  
* **Data Source:** Join Story\_Type with Parquet embeddings and visualize clusters.

**RQ10:** **Model Bias: The "Western" Gaze of OpenAI vs. Others**

* **Question:** Do Western-trained models (OpenAI) exhibit higher semantic drift (lower authenticity) for indigenous Bangladeshi cultures compared to models with more diverse training data (e.g., perhaps Qwen or Mistral)?  
* **Data Source:** Compare Cultural Accuracy mean scores by model across the Indigenous subset.

**RQ11:** **Semantic Drift in Moral Transgression**

* **Question:** For "Moral Transgression" stories, how closely does the consequence align with the specific culture's taboo system (e.g., eating forbidden food vs. social disrespect), or does it default to generic "sin"?  
* **Data Source:** Contextual & Temporal Appropriateness scores for this specific scenario.

---

### **Theme IV: Tradition & The "Uncanny"**

*Focus: The "folklore" of the artificial.*

**RQ12:** **The "Uncanny Valley" of Synthetic Tradition**

* **Question:** Is there a "sweet spot" of imperfection? Do stories with *slightly* lower linguistic scores actually rate higher in cultural authenticity because they avoid the "smooth," corporate polish of LLM speech?  
* **Data Source:** Non-linear correlation analysis between Linguistic and Cultural scores.

**RQ13:** **Temporal Anachronism in Indigenous Contexts**

* **Question:** To what extent do modern artifacts (phones, plastic, cars) "leak" into stories that are framed as "ancient legends", particularly in cultures the AI has less training data on?  
* **Data Source:** Contextual & Temporal Appropriateness low-score analysis.

**RQ14:** **Symbolic Resilience of the "Other"**

* **Question:** Which cultures act as "outliers" in the model? Which specific culture consistently receives the lowest Narrative & Symbolic Coherence scores, indicating it is the "least understood" by the AI?  
* **Data Source:** ANOVA test of scores grouped by Culture.

**RQ15:** **The Future of Fieldwork**

* **Question:** If AI can generate structurally coherent but culturally "hallucinated" variants of folklore, does this constitute a new form of "fakelore" (Dorson) that threatens to contaminate the digital preservation of real oral traditions?  
* **Data Source:** Synthesis of RQ1 and RQ2 results (High Coherence \+ Low Accuracy \= Dangerous Fakelore).

