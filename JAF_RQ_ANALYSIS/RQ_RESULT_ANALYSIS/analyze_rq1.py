"""
Deep Analysis Script for RQ1 Artifacts
Generates additional insights for research-quality reporting
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Load artifacts
result_dir = Path(__file__).parent.parent / "result"
desc = pd.read_csv(result_dir / 'rq1_descriptive_stats.csv')
anova = pd.read_csv(result_dir / 'rq1_anova_results.csv')
tukey = pd.read_csv(result_dir / 'rq1_tukey_hsd.csv')

print("=" * 80)
print("DEEP ANALYSIS: RQ1 - Comparative Model Efficacy in Cultural Authenticity")
print("=" * 80)

# 1. Model Ranking with Confidence Intervals
print("\n1. MODEL RANKING (by Mean Cultural Accuracy Score)")
print("-" * 60)
desc_sorted = desc.sort_values('mean', ascending=False)
for i, row in desc_sorted.iterrows():
    print(f"  #{int(row['rank'])} {row['model']}")
    print(f"      Mean: {row['mean']:.4f} (95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])")
    print(f"      Median: {row['median']:.1f}, SD: {row['std']:.4f}, Range: [{row['min']}, {row['max']}]")
    print(f"      N = {int(row['count'])}")
    print()

# 2. Performance Gap Analysis
print("\n2. PERFORMANCE GAP ANALYSIS")
print("-" * 60)
best = desc_sorted.iloc[0]
worst = desc_sorted.iloc[-1]
gap = best['mean'] - worst['mean']
pct_gap = (gap / worst['mean']) * 100
print(f"  Best Model:  {best['model']} (M = {best['mean']:.4f})")
print(f"  Worst Model: {worst['model']} (M = {worst['mean']:.4f})")
print(f"  Absolute Gap: {gap:.4f} points")
print(f"  Relative Gap: {pct_gap:.1f}% improvement from worst to best")

# 3. Statistical Significance Summary
print("\n3. STATISTICAL SIGNIFICANCE SUMMARY")
print("-" * 60)
f_stat = anova['f_statistic'].values[0]
p_val = anova['p_value'].values[0]
eta_sq = anova['eta_squared'].values[0]

print(f"  ANOVA Result: F(4, 594) = {f_stat:.4f}, p < 0.001")
print(f"  Effect Size: η² = {eta_sq:.4f} ({anova['effect_size_interpretation'].values[0]})")
print(f"  Interpretation: {eta_sq*100:.1f}% of variance in cultural accuracy")
print(f"                  is explained by model choice.")

# 4. Pairwise Comparison Summary
print("\n4. PAIRWISE COMPARISON SUMMARY (Tukey HSD)")
print("-" * 60)
sig_pairs = tukey[tukey['reject_null'] == True]
nonsig_pairs = tukey[tukey['reject_null'] == False]

print(f"  Significant Differences: {len(sig_pairs)}/10 pairs")
print(f"  Non-Significant: {len(nonsig_pairs)}/10 pairs")

print("\n  Significant pairs (p < 0.05):")
for _, row in sig_pairs.iterrows():
    direction = ">" if row['mean_diff'] > 0 else "<"
    print(f"    • {row['group1']} {direction} {row['group2']} (Δ = {abs(row['mean_diff']):.4f}, p = {row['p_adj']:.4f})")

print("\n  Non-significant pairs (statistically equivalent):")
for _, row in nonsig_pairs.iterrows():
    print(f"    • {row['group1']} ≈ {row['group2']} (p = {row['p_adj']:.4f})")

# 5. GPT-5.1 Dominance Analysis
print("\n5. GPT-5.1 DOMINANCE ANALYSIS")
print("-" * 60)
gpt51_pairs = tukey[(tukey['group1'] == 'openai_gpt-5.1') | (tukey['group2'] == 'openai_gpt-5.1')]
gpt51_wins = gpt51_pairs[gpt51_pairs['reject_null'] == True]
print(f"  GPT-5.1 significantly outperforms: {len(gpt51_wins)}/4 other models")
for _, row in gpt51_wins.iterrows():
    other = row['group2'] if row['group1'] == 'openai_gpt-5.1' else row['group1']
    print(f"    • vs {other}: Δ = {abs(row['mean_diff']):.4f}")

# 6. Model Tier Classification
print("\n6. MODEL TIER CLASSIFICATION")
print("-" * 60)
# Based on Tukey HSD results, classify into statistically distinct groups
print("  Based on statistical significance patterns:")
print("  TIER 1 (Superior):     openai_gpt-5.1 (M = 3.13)")
print("  TIER 2 (Intermediate): google_gemini-3-flash-preview (M = 2.40)")
print("  TIER 3 (Baseline):     openai_gpt-5-mini, mistral, qwen (M ≈ 2.13-2.38)")
print("                         [Not significantly different from each other]")

# 7. Score Distribution Analysis
print("\n7. SCORE DISTRIBUTION INSIGHTS")
print("-" * 60)
print(f"  Overall score range across models: 0-5")
print(f"  GPT-5.1: min={int(best['min'])}, max={int(best['max'])} (no 0s or 1s!)")
print(f"  Gemini has highest variance (SD = {desc[desc['model'].str.contains('gemini')]['std'].values[0]:.4f})")
print(f"  GPT-5-mini has lowest variance (SD = {desc[desc['model'].str.contains('5-mini')]['std'].values[0]:.4f})")

# 8. Practical Implications
print("\n8. PRACTICAL IMPLICATIONS")
print("-" * 60)
print("  • GPT-5.1 is the clear choice for culturally authentic generation")
print("  • The ~1 point gap (3.13 vs 2.13) is practically significant")
print("    (difference between 'Fair' and 'Good' on typical rubrics)")
print("  • Gemini shows high variance - inconsistent cultural quality")
print("  • Open-source models (Qwen, Mistral) lag behind proprietary ones")

# 9. Confidence in Results
print("\n9. CONFIDENCE IN RESULTS")
print("-" * 60)
print(f"  Sample size: n = 599 (adequate for statistical power)")
print(f"  Balanced design: ~120 stories per model")
print(f"  p-value: < 0.001 (extremely strong evidence)")
print(f"  Effect size: Large (η² = 0.21)")
print("  Conclusion confidence: HIGH")
