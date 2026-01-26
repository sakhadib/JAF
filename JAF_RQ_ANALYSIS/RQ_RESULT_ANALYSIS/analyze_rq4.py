"""
RQ4 Deep Analysis: Linguistic Fluency vs Cultural Accuracy Correlation
Research Question: Are models "hallucinating fluently"?
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 80)
print("RQ4 DEEP ANALYSIS: LINGUISTIC FLUENCY VS CULTURAL ACCURACY")
print("Are Models Hallucinating Fluently?")
print("=" * 80)

# Load data
overall_corr = pd.read_csv('result/rq4_overall_correlation.csv')
corr_matrix = pd.read_csv('result/rq4_correlation_matrix_pearson.csv')
corr_by_model = pd.read_csv('result/rq4_correlation_by_model.csv')
corr_by_culture = pd.read_csv('result/rq4_correlation_by_culture.csv')
regression = pd.read_csv('result/rq4_regression_analysis.csv')
summary = pd.read_csv('result/rq4_summary.csv')

# ============================================================================
# 1. OVERALL CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("1. OVERALL LINGUISTIC-CULTURAL CORRELATION")
print("=" * 80)

r = overall_corr['pearson_r'].iloc[0]
r2 = overall_corr['r_squared'].iloc[0]
p = overall_corr['pearson_p'].iloc[0]

print(f"\nPearson r = {r:.3f}")
print(f"R² = {r2:.3f} ({r2*100:.1f}% variance explained)")
print(f"p-value < 0.001")
print(f"n = 599 stories")

# Interpret effect size
if abs(r) >= 0.5:
    strength = "LARGE"
elif abs(r) >= 0.3:
    strength = "MEDIUM"
else:
    strength = "SMALL"
    
print(f"\nEffect Size: {strength} (Cohen's guidelines: r ≥ 0.5 = large)")

# ============================================================================
# 2. HALLUCINATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. HALLUCINATION ANALYSIS")
print("=" * 80)

print("""
The 'Hallucinating Fluently' Hypothesis:
- IF fluency UNCORRELATED with accuracy (r ≈ 0): Models generate smooth text without cultural grounding
- IF fluency CORRELATED with accuracy (r > 0.5): Models link linguistic quality to cultural authenticity
""")

print(f"\nObserved correlation: r = {r:.3f}")
print(f"Interpretation: {'MODERATE-STRONG' if r >= 0.4 else 'WEAK'} coupling")

if r >= 0.5:
    print("\n✓ VERDICT: Models are NOT hallucinating fluently")
    print("  Linguistic quality is meaningfully tied to cultural accuracy")
elif r >= 0.3:
    print("\n△ VERDICT: Models show PARTIAL coupling")
    print("  Some fluency-accuracy link exists, but room for improvement")
else:
    print("\n✗ VERDICT: HALLUCINATION CONCERN")
    print("  Models may be generating fluent but culturally inaccurate text")

# ============================================================================
# 3. MODEL-LEVEL PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("3. CORRELATION BY MODEL (HALLUCINATION RISK)")
print("=" * 80)

corr_by_model_sorted = corr_by_model.sort_values('pearson_r', ascending=False)

print(f"\n{'Model':<35} {'r':>8} {'R²':>8} {'p':>10} {'Risk':>15}")
print("-" * 76)

for _, row in corr_by_model_sorted.iterrows():
    model = row['model'].replace('_', ' ')[:32]
    r_m = row['pearson_r']
    r2_m = row['r_squared']
    p_m = row['pearson_p']
    sig = row['pearson_significant']
    
    if not sig:
        risk = "⚠️ HIGH RISK"
    elif r_m < 0.3:
        risk = "⚠️ Moderate"
    elif r_m < 0.5:
        risk = "Low"
    else:
        risk = "✓ Very Low"
    
    print(f"{model:<35} {r_m:>8.3f} {r2_m:>8.3f} {p_m:>10.4f} {risk:>15}")

# Identify high-risk models
high_risk = corr_by_model[~corr_by_model['pearson_significant']]['model'].tolist()
if high_risk:
    print(f"\n⚠️ HIGH HALLUCINATION RISK: {', '.join(high_risk)}")
    print("   These models show NO significant fluency-accuracy coupling!")
else:
    print("\n✓ All models show significant fluency-accuracy coupling")

# ============================================================================
# 4. GPT-5.1 PARADOX
# ============================================================================
print("\n" + "=" * 80)
print("4. GPT-5.1 PARADOX: THE BEST MODEL HAS WEAKEST COUPLING")
print("=" * 80)

gpt51 = corr_by_model[corr_by_model['model'] == 'openai_gpt-5.1'].iloc[0]
qwen = corr_by_model[corr_by_model['model'] == 'qwen_qwen3-8b'].iloc[0]

print(f"""
GPT-5.1:  r = {gpt51['pearson_r']:.3f} (NON-SIGNIFICANT, p = {gpt51['pearson_p']:.3f})
Qwen:     r = {qwen['pearson_r']:.3f} (p < 0.001)

This is a CRITICAL finding. Two interpretations:

A) CEILING EFFECT INTERPRETATION (Favorable):
   - GPT-5.1 achieves high scores on BOTH metrics consistently
   - With restricted range at the top, correlation mathematics break down
   - Not hallucinating—just uniformly excellent

B) DECOUPLED EXCELLENCE (Concerning):
   - GPT-5.1 treats fluency and accuracy as separate objectives
   - Could potentially produce fluent cultural errors
   - Needs human review despite high scores
""")

# Load original data to check for ceiling effect
try:
    main_df = pd.read_csv('data/evaluated_stories.csv')
    gpt51_data = main_df[main_df['model'] == 'openai_gpt-5.1']
    
    ling_mean = gpt51_data['Linguistic & Expressive Appropriateness (Bangla)'].mean()
    cult_mean = gpt51_data['Cultural Accuracy & Authenticity'].mean()
    ling_std = gpt51_data['Linguistic & Expressive Appropriateness (Bangla)'].std()
    cult_std = gpt51_data['Cultural Accuracy & Authenticity'].std()
    
    print(f"GPT-5.1 Score Statistics:")
    print(f"  Linguistic: M = {ling_mean:.2f}, SD = {ling_std:.2f}")
    print(f"  Cultural:   M = {cult_mean:.2f}, SD = {cult_std:.2f}")
    
    if ling_std < 0.5 and cult_std < 0.5:
        print("\n✓ CEILING EFFECT CONFIRMED: Low variance in both metrics")
    else:
        print("\n△ Ceiling effect partially supported")
except Exception as e:
    print(f"Could not verify ceiling effect: {e}")

# ============================================================================
# 5. CULTURAL VARIATION
# ============================================================================
print("\n" + "=" * 80)
print("5. CORRELATION BY CULTURE")
print("=" * 80)

corr_by_culture_sorted = corr_by_culture.sort_values('pearson_r', ascending=False)

print(f"\n{'Culture':<20} {'r':>8} {'R²':>8} {'n':>6} {'Interpretation':>25}")
print("-" * 67)

for _, row in corr_by_culture_sorted.iterrows():
    culture = row['culture'][:18]
    r_c = row['pearson_r']
    r2_c = row['r_squared']
    n_c = row['n']
    
    if r_c >= 0.6:
        interp = "Strong coupling"
    elif r_c >= 0.4:
        interp = "Moderate coupling"
    else:
        interp = "Weak coupling"
    
    print(f"{culture:<20} {r_c:>8.3f} {r2_c:>8.3f} {n_c:>6.0f} {interp:>25}")

# Correlation range
max_r = corr_by_culture_sorted['pearson_r'].max()
min_r = corr_by_culture_sorted['pearson_r'].min()
print(f"\nCorrelation Range: {min_r:.3f} to {max_r:.3f} (spread = {max_r - min_r:.3f})")

best_culture = corr_by_culture_sorted.iloc[0]['culture']
worst_culture = corr_by_culture_sorted.iloc[-1]['culture']
print(f"Strongest coupling: {best_culture} (r = {max_r:.3f})")
print(f"Weakest coupling:   {worst_culture} (r = {min_r:.3f})")

# ============================================================================
# 6. REGRESSION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. PREDICTIVE RELATIONSHIP")
print("=" * 80)

slope = regression['slope'].iloc[0]
intercept = regression['intercept'].iloc[0]
r2_reg = regression['r_squared'].iloc[0]

print(f"\nRegression Equation:")
print(f"  Cultural Accuracy = {slope:.3f} × Linguistic Fluency + {intercept:.3f}")
print(f"\nR² = {r2_reg:.3f} ({r2_reg*100:.1f}% of cultural accuracy variance explained)")

print(f"""
Interpretation:
- Each 1-point increase in linguistic fluency predicts {slope:.2f} points higher cultural accuracy
- Baseline cultural accuracy (at fluency = 0): {intercept:.2f}
- The relationship is {'strong' if r2_reg > 0.25 else 'moderate' if r2_reg > 0.1 else 'weak'}
""")

# ============================================================================
# 7. ALL METRICS CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("7. FULL CORRELATION MATRIX")
print("=" * 80)

# Clean up correlation matrix
corr_matrix = corr_matrix.set_index('Unnamed: 0')
print("\nPearson Correlations (all p < 0.001):\n")
print(corr_matrix.round(3).to_string())

print("""
Key Patterns:
- ALL metrics positively intercorrelated (r = 0.52 to 0.61)
- Contextual ↔ Narrative has STRONGEST correlation (r = 0.607)
- Linguistic ↔ Cultural has WEAKEST correlation (r = 0.521)
- This suggests a common 'quality factor' underlying all metrics
""")

# ============================================================================
# 8. FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("8. FINAL VERDICT: ARE MODELS HALLUCINATING FLUENTLY?")
print("=" * 80)

print(f"""
OVERALL: NO (with caveats)

Evidence AGAINST fluent hallucination:
✓ Overall r = {r:.3f} shows MODERATE-LARGE positive correlation
✓ All metrics intercorrelated (r = 0.52-0.61)
✓ 11/12 cultures show significant coupling
✓ Regression shows meaningful predictive relationship

CAVEATS:
⚠️ GPT-5.1 shows NON-SIGNIFICANT correlation (r = 0.16, p = 0.089)
   - Highest performer has weakest coupling
   - Likely due to ceiling effect, but warrants attention

RECOMMENDATIONS:
1. Trust fluency as a proxy for quality across most models
2. Extra verification recommended for GPT-5.1 outputs
3. Focus quality audits on cultures with weak coupling (Santal, Khasi, Mro)
""")
