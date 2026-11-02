# Model report

_Generated 2026-11-01 by `python run.py backtest`._

## Method
- **Walk-forward**: for each season N from 2010 to 2026, every model is trained only on
  seasons 2002..N-1 and then predicts every game of season N. Nothing from season N (or later) is used in training.
- **Features are pregame-only**: team EPA/success ratings use only games on earlier dates; QB ratings use only
  earlier dropbacks; Elo is the pregame value. No betting lines are used anywhere.
- **Spread and win% are consistent**: the production model predicts home margin, and
  win prob = Φ(margin / σ), with σ = 13.1 points fitted from training residuals.
- One caveat: in the backtest, the starting QB is the one who actually started (known at kickoff, after
  inactives). Live predictions made earlier in the week use the projected starter, so they can be a bit
  less accurate when a QB's status is uncertain.

## Out-of-sample results, 2010–2025 (4363 games)
Lower is better for Brier, log loss and spread MAE. A coin flip scores Brier 0.250 and log loss 0.693.

| model | description | games | brier | log_loss | accuracy | spread_mae |
|---|---|---|---|---|---|---|
| elo | Elo only (baseline, no fitting) | 4363 | 0.220 | 0.632 | 0.646 | 10.382 |
| ridge | **Production**: ridge regression on margin → Normal win prob | 4363 | 0.216 | 0.622 | 0.651 | 10.271 |
| ridge_no_qb | Ridge without the QB features (ablation) | 4363 | 0.219 | 0.628 | 0.645 | 10.328 |
| ridge_plus_weather | Ridge + wind/cold-dome features (ablation) | 4363 | 0.216 | 0.622 | 0.649 | 10.269 |
| logit | Logistic regression on win/loss (cross-check) | 4363 | 0.217 | 0.624 | 0.649 |  |
| gbm | Gradient boosting on margin | 4363 | 0.218 | 0.627 | 0.654 | 10.365 |

**Decisions**
- Ridge beats Elo on every metric, so it's the production model.
- Taking the QB features out (`ridge_no_qb`) clearly hurts, so they stay.
- Weather adds nothing (`ridge_plus_weather` vs `ridge`), so it is pruned.
- Gradient boosting does not beat ridge on Brier or log loss, so it is not used.
- The logistic regression lands within 0.017 (mean absolute difference) of the margin model's
  win probabilities, which confirms the Φ(margin/σ) conversion is sound.

## 2026 season so far (108 games, out of sample)
| model | games | brier | log_loss | accuracy | spread_mae |
|---|---|---|---|---|---|
| elo | 108 | 0.232 | 0.660 | 0.598 | 9.655 |
| ridge | 108 | 0.224 | 0.643 | 0.654 | 9.614 |
| logit | 108 | 0.224 | 0.643 | 0.636 |  |

## By season
| season | elo_brier | elo_acc | ridge_brier | ridge_acc | ridge_spread_mae | games |
|---|---|---|---|---|---|---|
| 2010 | 0.233 | 0.622 | 0.227 | 0.622 | 11.054 | 267 |
| 2011 | 0.213 | 0.659 | 0.208 | 0.667 | 10.565 | 267 |
| 2012 | 0.219 | 0.632 | 0.215 | 0.654 | 10.891 | 267 |
| 2013 | 0.216 | 0.647 | 0.214 | 0.643 | 9.968 | 267 |
| 2014 | 0.206 | 0.699 | 0.208 | 0.669 | 11.454 | 267 |
| 2015 | 0.224 | 0.674 | 0.222 | 0.644 | 10.102 | 267 |
| 2016 | 0.217 | 0.642 | 0.216 | 0.638 | 9.236 | 267 |
| 2017 | 0.217 | 0.674 | 0.212 | 0.652 | 10.483 | 267 |
| 2018 | 0.222 | 0.619 | 0.213 | 0.668 | 10.112 | 267 |
| 2019 | 0.222 | 0.643 | 0.220 | 0.639 | 10.431 | 267 |
| 2020 | 0.214 | 0.657 | 0.211 | 0.657 | 10.020 | 269 |
| 2021 | 0.231 | 0.602 | 0.224 | 0.630 | 10.927 | 285 |
| 2022 | 0.221 | 0.645 | 0.218 | 0.645 | 9.084 | 284 |
| 2023 | 0.233 | 0.604 | 0.225 | 0.646 | 10.262 | 285 |
| 2024 | 0.211 | 0.674 | 0.206 | 0.698 | 9.881 | 285 |
| 2025 | 0.222 | 0.644 | 0.217 | 0.644 | 9.943 | 285 |
| 2026 | 0.232 | 0.598 | 0.224 | 0.654 | 9.614 | 108 |

## Calibration (production model, 2010–2025)
When the model says the home team wins X% of the time, how often did it?

| home win prob | games | predicted | actual |
|---|---|---|---|
| (-0.001, 0.1] | 2 | 0.079 | 0.000 |
| (0.1, 0.2] | 46 | 0.167 | 0.196 |
| (0.2, 0.3] | 208 | 0.257 | 0.236 |
| (0.3, 0.4] | 486 | 0.356 | 0.343 |
| (0.4, 0.5] | 781 | 0.454 | 0.426 |
| (0.5, 0.6] | 972 | 0.549 | 0.536 |
| (0.6, 0.7] | 891 | 0.648 | 0.653 |
| (0.7, 0.8] | 700 | 0.745 | 0.759 |
| (0.8, 0.9] | 256 | 0.841 | 0.857 |
| (0.9, 1.0] | 21 | 0.919 | 0.952 |

## What drives the prediction
Ridge coefficients from training on 2002–2025, in points of home margin per 1 standard
deviation of each feature (the features overlap, so read these as rough importance, not causal effects).

|  | points per 1 SD |
|---|---|
| elo_diff | +3.45 |
| d_qb_change | +1.49 |
| d_success_off | +1.37 |
| d_qb_starter | +0.91 |
| d_rush_def | +0.79 |
| d_pass_def | +0.70 |
| d_rush_off | +0.70 |
| rest_diff | +0.56 |
| div_x_elo | -0.40 |
| d_pass_off | -0.36 |
| travel_diff | -0.19 |
| bye_diff | -0.14 |
| home | +0.13 |
| d_success_def | -0.05 |
