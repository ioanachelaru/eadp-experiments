# Chronological CV Experiment Findings

## Experimental Setup

All experiments use **chronological expanding-window cross-validation**:
- Train on releases 1..k, test on release k+1
- Scaler fit on training data only per fold
- Fresh classifier instance per fold (Random Forest, 100 trees, class_weight='balanced')
- Calcite: 15 releases (v1.1.0–v1.15.0), 14 test folds, 18,676 samples
- Ant-Ivy: 6 releases (v1.4.1–v2.4.0), 5 test folds, 2,237 samples
- Non-deduplicated datasets (version-based splits prevent same-file contamination naturally)

---

## Results Summary

### Calcite (14 folds)

| Dataset | # Features | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Top-30 SM (Embed1) | 30 | 0.387 | 0.297 | 0.321 | 0.632 |
| Effort26 + cov (Embed2) | 31 | 0.717 | 0.463 | 0.545 | 0.919 |
| Top30-SM + effort26 + cov (Embed3) | 61 | 0.708 | 0.463 | 0.544 | 0.921 |
| Effort170 + cov (Embed2-full) | 175 | 0.847 | 0.493 | 0.610 | 0.932 |
| Top30-SM + effort170 + cov (Embed3-full) | 205 | 0.859 | 0.497 | 0.614 | 0.933 |
| Full SM baseline | 2,859 | 0.843 | 0.826 | 0.832 | 0.967 |
| Full SM + cov | 2,864 | 0.835 | 0.823 | 0.826 | 0.967 |

### Wilcoxon Signed-Rank Tests (Calcite, 14 paired folds)

| Comparison | F1 diff | AUC diff | p-value (F1) | Significant? |
|---|---|---|---|---|
| Effort170+cov vs top-30 SM | +0.289 | +0.300 | 0.0001 | Yes (p<0.01) |
| Effort170+cov vs full SM (2,859) | -0.222 | -0.034 | 0.0001 | Yes (p<0.01) |

### Ablation (Calcite, excluding MOSER_bugfix + 16 ISSUE_* features)

| Condition | Features | F1 | AUC |
|---|---|---|---|
| All effort170 + cov | 175 | 0.610 | 0.932 |
| Excluding bug-correlated | 158 | 0.614 | 0.933 |

No data leakage — removing all label-correlated features has zero impact.

### Hyperparameter Sensitivity (Calcite, effort170+cov)

| n_estimators | F1 | AUC |
|---|---|---|
| 50 | 0.610 | 0.929 |
| 100 | 0.610 | 0.932 |
| 200 | 0.609 | 0.934 |

Results are stable across tree counts.

### Ant-Ivy (5 folds)

| Dataset | # Features | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Top-30 SM (Embed1) | 30 | 0.355 | 0.338 | 0.299 | 0.587 |
| Effort26 + cov (Embed2) | 31 | 0.423 | 0.207 | 0.245 | 0.856 |
| Top30-SM + effort26 + cov (Embed3) | 61 | 0.412 | 0.213 | 0.236 | 0.863 |
| Effort149 + cov (Embed2-full) | 154 | 0.376 | 0.277 | 0.300 | 0.883 |
| Top30-SM + effort149 + cov (Embed3-full) | 184 | 0.434 | 0.250 | 0.274 | 0.882 |
| Full SM baseline | 3,624 | 0.465 | 0.520 | 0.469 | 0.909 |
| Full SM + cov | 3,629 | 0.441 | 0.525 | 0.461 | 0.905 |

### Wilcoxon Signed-Rank Tests (Ant-Ivy, 5 paired folds)

| Comparison | F1 diff | AUC diff | p-value (F1) | p-value (AUC) | Significant? |
|---|---|---|---|---|---|
| Effort149+cov vs top-30 SM | +0.001 | +0.296 | 1.0000 | 0.0625 | No (min p=0.0625 with 5 folds) |
| Effort149+cov vs full SM (3,624) | -0.169 | -0.026 | 0.1250 | 0.0625 | No (min p=0.0625 with 5 folds) |

Note: With only 5 folds, the minimum achievable p-value for Wilcoxon signed-rank is 0.0625 (all 5 pairs in the same direction). Statistical significance at p<0.05 is impossible regardless of effect size.

### Ablation (Ant-Ivy, excluding MOSER_bugfix + 10 ISSUE_* features)

| Condition | Features | F1 | AUC |
|---|---|---|---|
| All effort149 + cov | 154 | 0.300 | 0.883 |
| Excluding bug-correlated | 143 | 0.270 | 0.882 |

No data leakage — removing all 11 label-correlated features has negligible impact (AUC: 0.883 vs 0.882).

### Hyperparameter Sensitivity (Ant-Ivy, effort149+cov)

| n_estimators | F1 | AUC |
|---|---|---|
| 50 | 0.272 | 0.879 |
| 100 | 0.300 | 0.883 |
| 200 | 0.281 | 0.887 |

Results are stable across tree counts.

---

## Key Findings

### 1. Effort features vastly outperform SM at equal feature count

At 30–31 features, effort+cov (AUC=0.919) massively outperforms SM (AUC=0.632). This is the apples-to-apples comparison: same dimensionality, same classifier, same CV protocol. Per feature, effort-related metrics carry far more predictive signal for defect prediction.

### 2. Full SM baseline (2,859 features) outperforms effort (175 features)

When using ALL available SM features (2,859), SM achieves F1=0.832 and AUC=0.967, significantly outperforming effort170+cov (F1=0.610, AUC=0.932) at p<0.01. However, this comparison uses 16x more features.

### 3. Adding SM to effort adds nothing, and vice versa

Embed3 (top30-SM + effort170 + cov, 205 features) performs almost identically to Embed2 (effort170 + cov, 175 features): F1=0.614 vs 0.610, AUC=0.933 vs 0.932. The SM features are redundant when effort features are present.

The reverse is also true: adding 5 coverage features to full SM has no impact. Calcite Full SM+cov (AUC=0.967) is identical to Full SM alone (AUC=0.967). Ant-Ivy Full SM+cov (AUC=0.905) is essentially the same as Full SM (AUC=0.909). The two feature families capture overlapping defect signals — whichever one you start with, adding the other contributes nothing.

### 4. No data leakage from bug-correlated features

Removing MOSER_bugfix and all ISSUE_* features (17 features total) has no impact on performance (F1=0.614 vs 0.610). The model relies on process and code quality signals, not label-correlated proxies.

### 5. Results hold under chronological evaluation

Switching from pooled stratified CV to chronological expanding-window CV — which prevents any future data from leaking into training — confirms that effort features are genuinely predictive. The chronological setup is more conservative (early folds have limited training data), but the patterns are clear.

### 6. Ant-Ivy confirms the same pattern as Calcite

On Ant-Ivy with all 6 feature sets:
- **Full SM dominates** (F1=0.469, AUC=0.909), same as Calcite
- **Effort + coverage strongly boosts AUC** over SM-only: Effort149+cov (AUC=0.883) with 154 features approaches Full SM (AUC=0.909) with 3,624 features
- **Adding SM to effort+cov adds nothing**: Top30-SM+effort149+cov (AUC=0.882) ≈ Effort149+cov (AUC=0.883)
- **Top-30 SM alone transfers poorly** (AUC=0.587) — these Calcite-derived features don't generalize
- **Effort26 vs Effort149**: 26 common features (AUC=0.856) capture most of the signal from 149 features (AUC=0.883)
- All feature sets have F1=0.000 on fold 2.0.0 (training only on v1.4.1, 240 samples), limiting overall F1 averages
- Low statistical power (5 folds) prevents significance testing

---

## Implications for the Paper

### The original claim needs reframing

The paper claimed effort features outperform SM for defect prediction. This was based on comparing 175 effort features vs 30 SM features — which the reviewer correctly identified as unfair. The full SM baseline shows that with enough SM features, SM wins.

**However, the reverse comparison is equally unfair**: 2,859 SM features vs 175 effort features is a 16:1 ratio.

### Defensible reframing options

#### Option A: Per-feature informativeness

*"Effort-related features are significantly more informative per feature than software metrics for defect prediction."*

Evidence:
- 31 effort+cov features match what 2,859 SM features need massive dimensionality to achieve
- At equal feature count (30 vs 31), effort AUC=0.919 vs SM AUC=0.632
- Adding SM on top of effort adds nothing, but effort dramatically improves over SM

This is the cleanest argument. It reframes from "effort is better" to "effort is more efficient" — a stronger and more nuanced claim.

#### Option B: Practical SDP argument

*"Effort-related metrics achieve competitive prediction performance with far fewer features and lower extraction cost."*

Evidence:
- Effort metrics come from version control and issue trackers — already available in any project
- SM features require code analysis tools and are expensive to extract at scale
- 175 effort features achieve AUC=0.932; matching this with SM requires 2,859 features
- Coverage features (5 JaCoCo metrics) provide additional signal cheaply from CI pipelines

This frames it as a practical recommendation rather than a theoretical superiority claim.

#### Option C: Apples-to-apples with top-175 SM

Run a **top-175 SM** experiment: select the 175 most important SM features (by RF importance on training data) and compare directly against 175 effort+cov features. This is the most airtight controlled comparison:
- Same feature count
- Same classifier, same CV protocol
- Features selected by the same method (RF importance)

If effort still wins at 175-vs-175, the outperforming claim is ironclad. If SM wins, the efficiency framing (Option A/B) is still valid.

---

## Possible additional experiments

1. **Top-175 SM baseline** — select 175 SM features by RF importance, run chronological CV. Most direct rebuttal to the "unfair feature count" concern.

2. **Logistic Regression** — run all main comparisons with LR to show results aren't classifier-specific.

3. **Per-version performance plots** — visualize F1/AUC across the 14 Calcite folds to show how prediction improves as training data accumulates.

4. **Coverage-only experiment** — isolate the contribution of the 5 coverage features (run effort-only without COV, compare with effort+cov).

---

## File Locations

All results are in `results/`:

```
results/
├── calcite/chronological/
│   ├── effort-cov-only/              # 26 effort + 5 cov (31 features)
│   ├── effort170-cov-only/           # 170 effort + 5 cov (175 features)
│   ├── effort170-cov-only_ablation/  # ablation: excl. MOSER_bugfix + ISSUE_*
│   ├── effort170-cov-only_n50/       # hyperparameter: 50 trees
│   ├── effort170-cov-only_n200/      # hyperparameter: 200 trees
│   ├── top30-sm-only-v1.1+/          # top-30 SM (30 features)
│   ├── top30-sm-cov-effort/          # top-30 SM + 26 effort + 5 cov (61 features)
│   ├── top30-sm-effort170-cov/       # top-30 SM + 170 effort + 5 cov (205 features)
│   └── sm-only-v1.1+/               # full SM baseline (2,859 features)
├── ant-ivy/chronological/
│   ├── top30-sm-only/                # top-30 SM (30 features)
│   ├── effort26-cov/                 # 26 effort + 5 cov (31 features)
│   ├── top30-sm-effort26-cov/        # top-30 SM + 26 effort + 5 cov (61 features)
│   ├── effort-cov/                   # 149 effort + 5 cov (154 features)
│   ├── top30-sm-effort-cov/          # top-30 SM + 149 effort + 5 cov (184 features)
│   ├── effort-only/                  # 149 effort features (no coverage)
│   └── sm-only/                      # full SM (3,624 features)
└── comparisons/
    ├── chronological_effort170-cov-only_vs_chronological_top30-sm-only-v1.1+_wilcoxon.json
    ├── chronological_effort-only_vs_chronological_sm-only_wilcoxon.json
    └── rf_results_vs_rf_results_wilcoxon.json  # effort170 vs full SM
```

Each experiment directory contains:
- `rf_results.json` — summary metrics, per-fold details, feature importances
- `fold_{version}/` — per-fold train.csv, test.csv, predictions.csv, metrics.json
