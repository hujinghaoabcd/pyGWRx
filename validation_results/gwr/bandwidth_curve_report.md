# Controlled GWR Bandwidth-Criterion Validation

All raw integer adaptive candidates `k=4..40` are archived. Strict comparisons
use criterion-specific validity domains so saturated or non-estimable boundary
candidates do not masquerade as optimal bandwidths.

- CV strict domain: candidates with finite CV from PyGWRx, mgwr, and GWmodel.
- AIC/BIC strict domain: k values below the near-saturated trace(S) boundary are excluded.
- AICc strict domain: candidates must be finite in all three implementations; this
  excludes k=4 where `n - 2 - trace(S) <= 0` and AICc is mathematically invalid.
- mgwr CV is converted from mean squared LOO error to SSE by multiplying by n=40.
- spgwr remains a semantic cross-check because its adaptive parameter is a continuous
  sample proportion q rather than an integer neighbour-order bandwidth.

## Validation domains

| Domain | First k | Last k | Candidates |
|---|---:|---:|---:|
| cv_common_finite | 6 | 40 | 35 |
| aic_nonsaturated | 5 | 40 | 36 |
| aicc_common_valid | 5 | 40 | 36 |
| bic_nonsaturated | 5 | 40 | 36 |
| spgwr_semantic | 5 | 40 | 36 |

## Pairwise curve comparisons

| Left | Right | Metric | Interpretation | Domain | n | Max abs diff | RMSE | Argmin left | Argmin right | Match |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| pyGWRx | mgwr | cv_sse | strict | cv_common_finite | 35 | 8.502968e-05 | 1.446513e-05 | 15 | 15 | yes |
| pyGWRx | GWmodel | cv_sse | strict | cv_common_finite | 35 | 8.519511e-05 | 1.448530e-05 | 15 | 15 | yes |
| mgwr | GWmodel | cv_sse | strict | cv_common_finite | 35 | 3.908847e-07 | 1.753943e-07 | 15 | 15 | yes |
| pyGWRx | mgwr | aic | strict | aic_nonsaturated | 36 | 3.647979e-05 | 1.239228e-05 | 5 | 5 | yes |
| pyGWRx | mgwr | aicc | strict | aicc_common_valid | 36 | 3.985457e-04 | 6.831670e-05 | 22 | 22 | yes |
| pyGWRx | GWmodel | aicc | strict | aicc_common_valid | 36 | 6.313387e-04 | 1.080019e-04 | 22 | 22 | yes |
| mgwr | GWmodel | aicc | strict | aicc_common_valid | 36 | 2.327930e-04 | 5.078080e-05 | 22 | 22 | yes |
| pyGWRx | mgwr | bic | strict | bic_nonsaturated | 36 | 3.577190e-05 | 1.041383e-05 | 5 | 5 | yes |
| pyGWRx | GWmodel | bic | definition_check | bic_nonsaturated | 36 | 4.368888e+01 | 4.368888e+01 | 5 | 5 | yes |
| mgwr | GWmodel | bic | definition_check | bic_nonsaturated | 36 | 4.368890e+01 | 4.368889e+01 | 5 | 5 | yes |
| pyGWRx | spgwr | cv_sse | different_adaptive_semantics | spgwr_semantic | 36 | 8.166870e+00 | 1.364562e+00 | 15 | 14 | no |

## Criterion minima on the controlled validation domains

| Implementation | Criterion | Argmin k | Domain | Note |
|---|---|---:|---|---|
| pyGWRx | cv_sse | 15 | cv_common_finite | strict three-way CV domain |
| mgwr | cv_sse | 15 | cv_common_finite | strict three-way CV domain |
| GWmodel | cv_sse | 15 | cv_common_finite | strict three-way CV domain |
| pyGWRx | aic | 5 | aic_nonsaturated | saturated k=4 excluded |
| mgwr | aic | 5 | aic_nonsaturated | saturated k=4 excluded |
| pyGWRx | aicc | 22 | aicc_common_valid | requires finite valid AICc |
| mgwr | aicc | 22 | aicc_common_valid | requires finite valid AICc |
| GWmodel | aicc | 22 | aicc_common_valid | requires finite valid AICc |
| pyGWRx | bic | 5 | bic_nonsaturated | saturated k=4 excluded |
| mgwr | bic | 5 | bic_nonsaturated | saturated k=4 excluded |
| GWmodel | bic | 5 | bic_nonsaturated | different BIC formula; diagnostic only |
| spgwr | cv_sse | 14 | spgwr_semantic | k is only q*n equivalence; spgwr optimizes continuous q |
| spgwr | aicc_like | 22 | spgwr_semantic | k is only q*n equivalence; spgwr optimizes continuous q |

## Low-bandwidth boundary finding

`k=4` is an essentially saturated smoother for this fixture (`trace(S)≈40` with n=40).
PyGWRx correctly returns infinite/invalid AICc there, while mgwr and GWmodel return
finite negative values. Those values are retained in the raw curve archive but are not
allowed to determine the validated AICc optimum. GWmodel also returns no finite CV at
k=4 or k=5; therefore the strict three-way CV domain begins at k=6.

## Interpretation rules

- `strict`: same integer-k bandwidth semantics and directly comparable criterion definition.
- `definition_check`: values are archived, but equality is not assumed until formulas are matched.
- `different_adaptive_semantics`: spgwr q=k/n is a sensitivity comparison, not an equality test.
