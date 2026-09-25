# M6 (first pass): EHOR estimator study (exploratory)

Tier 1 synthetic data, 300 replicates of 20,000 cases per condition. Truth is the realized EHOR of each replicate, so bias and coverage are design-based. Naive comparator: intercepted errors over intercepted plus 30% of missed errors (an assumed discovery rate). Stratification is by displayed confidence, oversampling low-confidence outputs 6-fold against high. Reviewer settings are assumptions (design section 11).

| Condition | Mean true EHOR | Naive bias | Naive RMSE | Audit % | Stratified | HT bias | HT RMSE | 95% CI coverage |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.938 | +0.042 | 0.044 | 1% | no | -0.0695 | 0.3044 | 0.13 |
| baseline | 0.938 | +0.042 | 0.044 | 1% | yes | -0.0169 | 0.2160 | 0.14 |
| baseline | 0.938 | +0.042 | 0.044 | 5% | no | -0.0057 | 0.0676 | 0.59 |
| baseline | 0.938 | +0.042 | 0.044 | 5% | yes | +0.0058 | 0.0786 | 0.51 |
| baseline | 0.938 | +0.042 | 0.044 | 20% | no | -0.0022 | 0.0298 | 0.91 |
| baseline | 0.938 | +0.042 | 0.044 | 20% | yes | +0.0009 | 0.0366 | 0.79 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 1% | no | -0.0842 | 0.3465 | 0.32 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 1% | yes | -0.0080 | 0.2541 | 0.32 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 5% | no | -0.0026 | 0.0993 | 0.85 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 5% | yes | +0.0019 | 0.1118 | 0.76 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 20% | no | -0.0015 | 0.0441 | 0.95 |
| SC10 low (+10 pct) | 0.846 | +0.102 | 0.103 | 20% | yes | +0.0023 | 0.0474 | 0.91 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 1% | no | -0.0730 | 0.3681 | 0.44 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 1% | yes | +0.0001 | 0.2914 | 0.55 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 5% | no | +0.0059 | 0.1194 | 0.90 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 5% | yes | -0.0072 | 0.1190 | 0.89 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 20% | no | +0.0004 | 0.0531 | 0.94 |
| SC10 medium (+25 pct) | 0.707 | +0.182 | 0.183 | 20% | yes | -0.0028 | 0.0688 | 0.88 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 1% | no | -0.0437 | 0.3682 | 0.51 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 1% | yes | -0.0229 | 0.3363 | 0.55 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 5% | no | -0.0153 | 0.1270 | 0.94 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 5% | yes | +0.0098 | 0.1316 | 0.90 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 20% | no | +0.0026 | 0.0645 | 0.92 |
| SC10 high (+40 pct) | 0.562 | +0.248 | 0.248 | 20% | yes | -0.0046 | 0.0640 | 0.95 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 1% | no | -0.0556 | 0.3179 | 0.31 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 1% | yes | -0.0044 | 0.2505 | 0.38 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 5% | no | -0.0046 | 0.1014 | 0.89 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 5% | yes | +0.0059 | 0.0989 | 0.81 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 20% | no | -0.0013 | 0.0426 | 0.94 |
| SC11 low (90 pct coverage) | 0.844 | +0.103 | 0.104 | 20% | yes | +0.0027 | 0.0527 | 0.85 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 1% | no | -0.0382 | 0.3546 | 0.48 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 1% | yes | -0.0055 | 0.3198 | 0.56 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 5% | no | -0.0068 | 0.1407 | 0.88 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 5% | yes | +0.0085 | 0.1348 | 0.84 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 20% | no | -0.0050 | 0.0548 | 0.95 |
| SC11 medium (70 pct) | 0.657 | +0.207 | 0.208 | 20% | yes | -0.0067 | 0.0621 | 0.92 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 1% | no | -0.0457 | 0.3442 | 0.45 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 1% | yes | -0.0067 | 0.3016 | 0.62 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 5% | no | +0.0030 | 0.1446 | 0.87 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 5% | yes | -0.0089 | 0.1343 | 0.86 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 20% | no | +0.0048 | 0.0585 | 0.96 |
| SC11 high (40 pct) | 0.374 | +0.291 | 0.291 | 20% | yes | +0.0035 | 0.0671 | 0.90 |

IIR (5 percent stratified audit) bias, worst condition: 0.0007.
