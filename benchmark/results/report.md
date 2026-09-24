# logitly calibration benchmark

Generated 2026-09-24T12:17:49+00:00 by `benchmark/run.py`. Do not edit by hand.

| setting | value |
|---|---|
| models | `qwen3.5-2b` |
| test examples | 1000 |
| calibration pool | 512 |
| calibration sizes | 8, 16, 32, 64, 256 |
| seeds | 0, 1, 2, 3, 4 |
| ECE bins | 15 |
| calibration draw | natural mix |

## Summary

Raw vs. calibrated on 256 examples, averaged over 5 seeds.

| dataset | model | accuracy | NLL | ECE | Brier | changed |
|---|---|---|---|---|---|---|
| sst2 | qwen3.5-2b | 0.881 → 0.906 (+0.025) ✓ | 0.269 → 0.252 (-0.017) ✓ | 0.028 → 0.026 (-0.002) ✓ | 0.159 → 0.143 (-0.016) ✓ | 7.7% |
| ag_news | qwen3.5-2b | 0.835 → 0.864 (+0.029) ✓ | 0.607 → 0.405 (-0.202) ✓ | 0.089 → 0.022 (-0.067) ✓ | 0.260 → 0.207 (-0.053) ✓ | 8.7% |
| trec | qwen3.5-2b | 0.766 → 0.802 (+0.036) ✓ | 0.779 → 0.657 (-0.123) ✓ | 0.067 → 0.057 (-0.011) ✓ | 0.360 → 0.298 (-0.062) ✓ | 16.1% |
| emotion | qwen3.5-2b | 0.562 → 0.566 (+0.004) ✓ | 1.516 → 1.190 (-0.326) ✓ | 0.218 → 0.048 (-0.170) ✓ | 0.656 → 0.570 (-0.086) ✓ | 11.4% |

## Per dataset

- [sst2](sst2/report.md)
- [ag_news](ag_news/report.md)
- [trec](trec/report.md)
- [emotion](emotion/report.md)
