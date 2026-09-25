# M6 (first pass): EHOR estimator study (exploratory)

Tier 1 synthetic data, 300 replicates of 20,000 cases per condition. Truth is the realized EHOR of each replicate, so bias and coverage are design-based. Naive comparator: intercepted errors over intercepted plus 30% of missed errors (an assumed discovery rate). Stratification is by displayed confidence, oversampling low-confidence outputs 6-fold against high. Reviewer settings are assumptions (design section 11).

| Condition | Mean true EHOR | Naive bias | Naive RMSE | Audit % | Stratified | HT bias | HT RMSE | 95% CI coverage |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.923 | +0.052 | 0.053 | 1% | no | -0.0022 | 0.0809 | 0.62 |
| baseline | 0.923 | +0.052 | 0.053 | 1% | yes | +0.0023 | 0.0942 | 0.57 |
| baseline | 0.923 | +0.052 | 0.053 | 5% | no | +0.0010 | 0.0359 | 0.89 |
| baseline | 0.923 | +0.052 | 0.053 | 5% | yes | +0.0007 | 0.0401 | 0.83 |
| baseline | 0.923 | +0.052 | 0.053 | 20% | no | +0.0011 | 0.0152 | 0.93 |
| baseline | 0.923 | +0.052 | 0.053 | 20% | yes | +0.0006 | 0.0192 | 0.91 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 1% | no | +0.0007 | 0.1222 | 0.82 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 1% | yes | +0.0049 | 0.1094 | 0.80 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 5% | no | +0.0034 | 0.0479 | 0.92 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 5% | yes | +0.0014 | 0.0526 | 0.89 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 20% | no | -0.0012 | 0.0222 | 0.94 |
| SC10 low (+10 pct) | 0.831 | +0.112 | 0.112 | 20% | yes | -0.0010 | 0.0209 | 0.98 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 1% | no | -0.0026 | 0.1353 | 0.93 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 1% | yes | +0.0012 | 0.1341 | 0.86 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 5% | no | +0.0045 | 0.0619 | 0.93 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 5% | yes | -0.0011 | 0.0574 | 0.96 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 20% | no | +0.0003 | 0.0269 | 0.95 |
| SC10 medium (+25 pct) | 0.693 | +0.190 | 0.190 | 20% | yes | +0.0015 | 0.0310 | 0.91 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 1% | no | -0.0114 | 0.1522 | 0.91 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 1% | yes | -0.0021 | 0.1458 | 0.88 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 5% | no | -0.0059 | 0.0569 | 0.98 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 5% | yes | +0.0024 | 0.0644 | 0.92 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 20% | no | +0.0025 | 0.0288 | 0.95 |
| SC10 high (+40 pct) | 0.554 | +0.251 | 0.251 | 20% | yes | -0.0009 | 0.0302 | 0.92 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 1% | no | +0.0005 | 0.1119 | 0.85 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 1% | yes | -0.0109 | 0.1157 | 0.83 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 5% | no | -0.0014 | 0.0471 | 0.94 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 5% | yes | +0.0005 | 0.0539 | 0.88 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 20% | no | -0.0010 | 0.0225 | 0.95 |
| SC11 low (90 pct coverage) | 0.831 | +0.112 | 0.112 | 20% | yes | +0.0019 | 0.0236 | 0.94 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 1% | no | -0.0158 | 0.1508 | 0.91 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 1% | yes | +0.0095 | 0.1459 | 0.84 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 5% | no | -0.0068 | 0.0594 | 0.96 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 5% | yes | +0.0096 | 0.0609 | 0.94 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 20% | no | -0.0010 | 0.0278 | 0.96 |
| SC11 medium (70 pct) | 0.646 | +0.213 | 0.213 | 20% | yes | -0.0041 | 0.0279 | 0.96 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 1% | no | -0.0056 | 0.1472 | 0.91 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 1% | yes | +0.0052 | 0.1350 | 0.90 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 5% | no | -0.0034 | 0.0586 | 0.94 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 5% | yes | -0.0013 | 0.0602 | 0.93 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 20% | no | +0.0015 | 0.0272 | 0.94 |
| SC11 high (40 pct) | 0.369 | +0.292 | 0.292 | 20% | yes | -0.0002 | 0.0295 | 0.95 |

IIR (5 percent stratified audit) bias, worst condition: 0.0007.
