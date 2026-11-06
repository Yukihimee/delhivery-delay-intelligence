# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project for the Artificial Intelligence and Machine Learning Certification Program, E&ICT Academy, IIT Roorkee.

The data is synthetic. Every row comes from `generate_data.py`, and no real Delhivery shipment records are used. The repository answers three operations questions with one connected pipeline:

| Question | Where it is answered |
|---|---|
| How long will a parcel really take? | Part B, delivery-hours regression (`outputs/part_b_*.csv`) |
| Which parcels will miss their promise? | Part C, delay classifier and alert threshold (`outputs/part_c_*.csv`, `models/`) |
| Which hubs need which playbook? | Part D, hub segments (`outputs/hub_segments.csv`, `outputs/segment_profile.csv`) |

`predict_delay.py` combines Part C's classifier with Part D's segments to turn a batch of new parcels into an action list (`outputs/new_shipment_actions.csv`).

```
README.md  requirements.txt  generate_data.py
data/      hubs.csv, raw_shipments.csv, new_shipments.csv (generator output), modelling_table.csv (Part A)
src/       shipment_prep.py: shared constants, clean_shipments, build_features, the train/validation/test split
part_a_clean_eda.py  part_b_regression.py  part_c_classification.py
part_d_segments.py   package_model.py      predict_delay.py
models/    delay_model.pkl, delay_threshold.csv, feature_order.csv, impute_values.csv
outputs/   every result CSV named in the brief
```

## How to run

These commands start from a fresh clone. The pipeline was tested with Python 3.12. The pinned versions also install on Python 3.10 to 3.13.

```bash
git clone https://github.com/Yukihimee/delhivery-delay-intelligence.git
cd delhivery-delay-intelligence
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py              # prints 24 1224 48 and rewrites data/ with identical contents
python part_a_clean_eda.py           # data/modelling_table.csv and outputs/part_a_*.csv
python part_b_regression.py          # outputs/part_b_*.csv
python part_c_classification.py      # outputs/part_c_*.csv (the slowest step: 135 + 45 search fits)
python part_d_segments.py            # outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv, ...
python package_model.py              # regenerates models/delay_model.pkl and the other models/ files
python predict_delay.py              # outputs/new_shipment_actions.csv
```

- Run the scripts in this order because each one reads files from an earlier step. `package_model.py` reads `outputs/part_c_final_config.csv`, and `predict_delay.py` reads `models/` and `outputs/hub_segments.csv`.
- `python package_model.py` regenerates `models/delay_model.pkl`. The committed pickle loads reliably only on the scikit-learn version pinned in `requirements.txt` (1.5.2). If your installed versions differ, run `package_model.py` again instead of loading the committed pickle.
- Run `generate_data.py` from the repository root, because it is kept verbatim and writes to `./data`. Every other script finds its files relative to the repository root, so you can start it from any working directory.
- Every `random_state` is 42. Re-running the full pipeline on the same platform reproduced every committed data, output and model file byte for byte.
- Part C may print a harmless `OptimizeWarning` about `iprint` (from newer SciPy releases) and a NumPy `RuntimeWarning` from the search results table. On some Windows machines, joblib also warns that it cannot count physical cores; set `LOKY_MAX_CPU_COUNT` to silence it. None of these warnings changes any result.

## Data findings

**Cleaning** (`outputs/part_a_checks.csv`):

- The export has 1224 rows. Removing 24 exact duplicates leaves 1200 shipments, and all 1200 survive the hub merge.
- Missing values were filled with medians from the full historical table:
  - `distance_km`: 36 gaps, filled with 314.95 km.
  - `parcel_weight_kg`: 24 gaps plus 5 scanner-fault sentinels of 999 kg, filled with 1.83 kg.
  - `declared_value_inr`: 12 gaps, filled with INR 1,121.50.
- 18 blank payment modes became `Unknown`.
- `service_type` had 12 spellings before cleaning and 3 after.
- 278 parcels missed their promise, a delay rate of 0.2317.
- The 1.5 × IQR rule on `delivery_hours` (Q1 43.255, Q3 72.5625, bounds -0.7063 / 116.5238) flags 1 parcel: SHP10691, an Economy parcel that took 117.18 hours over 721.8 km in rain during the sale. I kept it because it is a plausible long delivery, not a data error.

1. **Express is delayed most often.** 119 of its 243 parcels were late, a delay rate of 0.4897. Standard's rate is 0.2056 and Economy's is 0.0857. The mean Express delivery time, 36.8056 hours, is already above Express's 36-hour promise, so the tightest promise has no slack.
2. **Delay rate rises with tier.** It is 0.04 in Tier 1 (16 of 400 parcels), 0.215 in Tier 2 (86 of 400) and 0.44 in Tier 3 (176 of 400). Tier 3 hubs miss eleven times as often as Tier 1 hubs.
3. **Delay rate rises with every distance band** (`outputs/part_a_delay_by_group.csv`). It is 0.0219 for 0-200 km (6 of 274), 0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km (75 of 88). Beyond 600 km, more than 8 in 10 parcels are late.
4. **Three columns correlate most strongly with `delivery_hours`** (`outputs/part_a_correlations.csv`, ranked by absolute value): `promised_hours` +0.8372, `distance_km` +0.4013 and `n_stops` +0.3247. `promised_hours` stands in for the service type. It sets the level of delivery hours, because Economy is slow by design, but it does not decide whether a parcel is late. Distance and route stops are the operational drivers. All five bookkeeping columns stay within ±0.03; the largest is `sticker_serial_mod` at 0.0293.
5. **`declared_value_inr` was log-transformed because it is right-skewed.** In the median-filled column, a long tail of expensive parcels pulls the mean (INR 1,533.05) well above the median (INR 1,121.50). After the natural log, the mean (7.0277) and median (7.0224) almost coincide, so a few very expensive parcels no longer dominate a scaled or linear model.

## Model results (regression)

**Split and scaling.** The data is split 840 / 180 / 180 (`outputs/part_b_checks.csv`). The scaler was fitted on the training rows only. The largest absolute column mean of the scaled training rows is 0.0 to 4 decimals, while the scaled test rows show 0.2187. This shows that the test rows did not influence the scaler.

Validation results (`outputs/part_b_metrics.csv`):

| model | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|
| mean_baseline | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| linear_regression | 3.9222 | 23.9235 | **4.8912** | 0.9313 |
| decision_tree (depth 6) | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| random_forest | 4.2882 | 30.355 | 5.5095 | 0.9128 |
| gradient_boosting | 3.9166 | 24.1722 | 4.9165 | 0.9306 |

- **Baseline.** Predicting the training mean for every parcel gives R² -0.0212. It explains none of the variation and does slightly worse than predicting the validation set's own mean. Any useful model must beat its RMSE of 18.8552 hours.
- **Validation winner.** Linear regression wins with an RMSE of 4.8912 hours (R² 0.9313). It is only 0.0253 hours ahead of gradient boosting (4.9165). Linear regression does well because delivery time here is built mostly from additive effects: distance, stops, service type, rain, sale, COD, weekend and evening dispatch.
- **Tree depth** (`outputs/part_b_tree_depth.csv`):
  - Depths 1-3 have high bias. Both errors are high and close together. At depth 1, train RMSE is 11.6334 and validation RMSE is 12.1324, a gap of only 0.499.
  - Depth 6 has the lowest validation RMSE, 6.2796.
  - Depths 8, 12 and unlimited have high variance. Train RMSE keeps falling (3.336, then 1.3613, then 0.0) while validation RMSE rises (6.7901, then 7.354, then 7.8368).
  - The unlimited tree memorises the training rows (train RMSE 0.0) and misses new parcels by 7.8368 hours.
- **Learning rate** (`outputs/part_b_gb_learning_rate.csv`):
  - 0.1 is best, with validation R² 0.9306 and RMSE 4.9165.
  - 0.01 is too slow. Each tree adds only 1% of its correction, so 100 trees stop far short of the target and the model underfits (R² 0.7127, RMSE 10.0018).
  - 1.0 overshoots. Each tree applies its full correction and chases noise in the previous trees' residuals (R² 0.844, RMSE 7.3701).
- **Cross-validation** (`outputs/part_b_cv.csv`) used the 1,020 training and validation rows and one shared `KFold(5, shuffle=True, random_state=42)`. Gradient boosting reaches mean R² 0.9353 (std 0.0045) and beats linear regression at 0.9261 (std 0.0048). This is the opposite of the validation ranking.
- **Final model.** Under the brief's rule, the final model is the validation winner, linear regression. Scored once on the test rows, it reaches RMSE 4.5235 hours, MAE 3.5671 and R² 0.9493.
- **Which model I would ship.** Because the two rankings disagree, I would ship gradient boosting:
  - The cross-validated comparison averages five folds over 1,020 rows instead of relying on one 180-row split.
  - Its R² lead of 0.0092 is about twice the fold-to-fold standard deviation. The validation gap of 0.0253 hours of RMSE is practically a tie.
  - Boosting can learn interactions that linear regression cannot, such as Express parcels during the sale and long lanes in rain. Those are exactly the situations planners worry about.
  - I deliberately did not measure its test score. I would confirm it on the next month of delivered parcels before replacing the linear model.

## Model results (classification)

The stratified split is 840 / 180 / 180 parcels, with 195 / 42 / 41 delayed parcels.

Validation results at the default 0.5 rule (`outputs/part_c_models.csv`):

| model | TN | FP | FN | TP | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| blind_all_on_time | 138 | 0 | 42 | 0 | 0.7667 | 0.0 | 0.0 | 0.0 |
| logistic_regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.747 |
| random_forest_default | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

- **The blind baseline is useless.** Calling every parcel on time scores accuracy 0.7667 but catches 0 of the 42 delayed parcels (recall 0.0). Accuracy rewards the on-time majority, so it looks respectable for a model that never finds a delayed parcel, which is the minority operations needs to catch. Every trained model beats it on accuracy (0.8833 to 0.9056). More importantly, each one catches 31 to 35 of the 42 delayed parcels.
- **Signal versus noise** (`outputs/part_c_feature_scores.csv`):
  - The five bookkeeping columns together take 0.2245 of the default forest's importance (0.046 + 0.0374 + 0.0476 + 0.0499 + 0.0436).
  - Each of them outranks real drivers such as `is_cod` (0.0106) and `is_weekend` (0.0087). Impurity importance rewards high-cardinality columns that trees can split many times, even when those splits only fit noise.
  - The F-test is not fooled. All five noise columns fall outside the top 7 by F-score, at ranks 8, 12, 14, 16 and 17, with F-scores of 4.5847 and below.
  - The top 7 are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`. All of them score at least 36.7281.
- **Number of columns** (`outputs/part_c_selectk.csv`). Validation F1 is 0.6988 at k = 5, peaks at 0.8333 at k = 7, then reaches 0.7857 at k = 10, 0.8049 at k = 12 and 0.7848 at k = 17. With five columns, two real drivers (`rain_flag` and `sale_event_flag`) drop out. Beyond seven, the forest gets weak and pure-noise columns that it partly fits.
- **Tuning** (`outputs/part_c_tuning.csv`):

  | search | fits | best parameters | CV accuracy | validation F1 |
  |---|---|---|---|---|
  | Grid search | 27 candidates × 5 folds = 135 | max_depth 6, min_samples_leaf 1, 100 trees | 0.8952 | 0.775 |
  | Randomized search | 9 × 5 = 45 | max_depth 6, min_samples_leaf 1, 50 trees | 0.8905 | 0.7848 |
  | Untuned forest, same 7 columns | 1 | defaults | (not cross-validated) | 0.8333 |

  - Neither search beat the untuned forest on validation, so `default_forest_same_7_features` became the final configuration (`outputs/part_c_final_config.csv`).
  - The searches optimise cross-validated accuracy, not F1, and the shallower trees they prefer flag fewer parcels.
  - More importantly, the validation set holds only 42 delayed parcels. One parcel moves F1 by roughly 0.01 to 0.015, so the whole spread is a handful of parcels. On a 180-row validation set, tuning gains are within noise. This result does not prove that tuning hurts.
- **Alert threshold** (`outputs/part_c_thresholds.csv`). The operations rule picks the highest threshold with validation recall of at least 0.80, which is **0.5**. At 0.5, recall is 0.8333 (35 of 42) and precision is 0.7955.
  - Raising the threshold to 0.6 lifts precision to 0.8889, but recall falls to 0.7619 (32 of 42), which breaks the rule.
  - Lowering it to 0.4 or 0.3 buys recall of 0.881 or 0.9286 at precision 0.74 or 0.6964, which means more unnecessary calls.
  - The 0.5 row of this table (TP 35, FP 9, F1 0.814) differs slightly from the forest's own `predict`, which gives validation F1 0.8333 in the tuning table. The operations rule flags parcels at probability ≥ 0.5, while `predict` requires a probability strictly above 0.5. Two on-time parcels sit at exactly 0.5.
- **Test result** (`outputs/part_c_final_test.csv`, evaluated once). At threshold 0.5, the final model reaches precision 0.6889, recall 0.7561 and F1 0.7209 (TP 31, FP 14, FN 10, TN 125).
  - Test recall is below both the validation recall of 0.8333 and the 0.80 target. The model caught 31 of 41 delayed parcels, where the validation rate would have meant about 34.
  - With only 41 delayed test parcels, each one moves recall by about 0.024, so the gap is about three parcels. Part of it is chance, and part is the optimism of choosing the threshold on the validation rows it was then measured on.
  - I did not re-tune after seeing this number. Operations should plan for recall in the mid-0.7s rather than 0.83.

## Operations recommendation

**Escalate segment.** Nine hubs belong to this segment, with a mean delay rate of 0.46 (`outputs/segment_profile.csv`, `outputs/hub_profile.csv`):

| hub | city | tier label | delay rate |
|---|---|---|---|
| HUB13 | Nagpur | 2 | 0.46 |
| HUB15 | Kochi | 2 | 0.36 |
| HUB17 | Patna | 3 | 0.46 |
| HUB18 | Ranchi | 3 | 0.38 |
| HUB19 | Guwahati | 3 | 0.62 |
| HUB20 | Raipur | 3 | 0.48 |
| HUB21 | Bhubaneswar | 3 | 0.4 |
| HUB23 | Siliguri | 3 | 0.44 |
| HUB24 | Agartala | 3 | 0.54 |

| segment | hubs | mean hours | delay rate | mean distance (km) | mean stops |
|---|---|---|---|---|---|
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.155 | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.46 | 470.4757 | 84.3444 |

**Four hub tier labels are misleading** (`outputs/tier_vs_segment_mismatch.csv`):

- **HUB07 Kolkata**: Tier 1, but behaves as Watch. Delay rate 0.14, 304.736 km, 70.76 stops: a regional profile, not a metro one.
- **HUB13 Nagpur**: Tier 2, but behaves as Escalate. Delay rate 0.46, 473.32 km, 83.92 stops.
- **HUB15 Kochi**: Tier 2, but behaves as Escalate. Delay rate 0.36, 466.514 km, 84.1 stops.
- **HUB22 Dehradun**: Tier 3, but behaves as Watch. Delay rate 0.2, 316.738 km, 73.28 stops.

The tier label follows hub size, since daily capacity falls steadily from Tier 1 to Tier 3 in `data/hubs.csv`. It does not describe how a hub actually runs.

- Use the segment, not the tier, to choose a hub's playbook, staffing and escalation route. Keep the tier label for capacity planning only.
- `hub_tier` is also a model input, so check the classifier on these four hubs at the next retraining. Replacing `hub_tier` with the segment or a trailing hub delay rate is the obvious candidate.

**Alert threshold: 0.5.** On the untouched test set (`outputs/part_c_final_test.csv`):

- Precision is 0.6889, so about 7 of every 10 flagged parcels really are delayed (31 of 45 flagged).
- Recall is 0.7561, so about 7.6 of every 10 delayed parcels are caught (31 of 41). Roughly 2 to 3 of every 10 still slip through.

**New parcels** (`outputs/new_shipment_actions.csv`). The model flags 8 of the 48 parcels:

- **6 need a hub-manager escalation.** All are at Escalate hubs: SHP90002 (HUB24, probability 0.74), SHP90003 (HUB18, 0.88), SHP90011 (HUB17, 0.89), SHP90025 (HUB20, 0.64), SHP90041 (HUB21, 0.85) and SHP90046 (HUB21, 0.53).
- **2 get a proactive customer call.** Both are at Watch hubs: SHP90005 (HUB22 Dehradun, 0.73) and SHP90007 (HUB09 Jaipur, 0.53).
- **The other 40 parcels need no action.**

SHP90005 shows why the segment matters. Its Tier 3 label would have sent it to escalation, but Dehradun behaves like a Watch hub, so a customer call is the proportionate response.

**Why K = 3** (`outputs/part_d_separation.csv`). The across-to-within distance ratio is:

| K | ratio | cluster sizes |
|---|---|---|
| 2 | 2.894 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

The ratio keeps rising because clusters get smaller, but the big gain comes from moving from K = 2 to K = 3. K = 4 and K = 5 both create clusters of fewer than 6 hubs (5 and 4 hubs at K = 4; 4, 3 and 2 at K = 5), which are too small to carry their own playbook. K = 3 matches the three playbooks operations runs, and every segment has at least 7 hubs.

## Production plan

**(a) Production-readiness characteristics and where they live in this repository**

| characteristic | concrete item |
|---|---|
| Accuracy | Held-out evidence is in `outputs/part_b_metrics.csv` (test R² 0.9493) and `outputs/part_c_final_test.csv` (test F1 0.7209). The alert threshold comes from the recall rule in `outputs/part_c_thresholds.csv`, not from accuracy. |
| Reliability | `clean_shipments` repairs casing, blank payment modes, rupee-formatted values and 999 kg sentinels, and fills gaps with stored medians. A new parcel with missing fields is still scored: `new_shipments.csv` contains blanks, while `new_shipment_actions.csv` has none. `build_features` uses a validated many-to-one left merge, so the hub join can never duplicate parcels. |
| Reproducibility | `requirements.txt` pins pandas, numpy, scikit-learn and joblib, and every `random_state` is 42. `package_model.py` rebuilds the model from `outputs/part_c_final_config.csv`. `outputs/part_d_reload_check.csv` shows that the reloaded pickle reproduces the in-memory probabilities (max difference 0.0), and two runs of `predict_delay.py` give byte-identical files. |
| Performance (speed) | The deployed model is a single Pipeline: `SelectKBest` with k = 7, then a 100-tree forest. `predict_delay.py` scores a whole batch with one vectorised `predict_proba` call and runs in seconds on a laptop CPU, with no GPU or external service. |
| Maintainability | One shared module, `src/shipment_prep.py`, holds `PROMISED_HOURS`, `FEATURES`, `NOISE_COLUMNS`, `clean_shipments`, `build_features` and the split. Each Part is one script with one job. |
| Consistency | `predict_delay.py` calls the same `clean_shipments`, with the training medians from `models/impute_values.csv`, and the same `build_features` as training. It orders columns by `models/feature_order.csv`. The selector travels inside the pickled Pipeline, so scoring cannot skip a training step. |
| Monitorability | Every scored parcel is logged with its probability, flag, segment and action in `outputs/new_shipment_actions.csv`, ready to join with the delivered outcome. The triggers below define what to watch. |

**(b) What to monitor and when to act**

| quantity | reference value | trigger |
|---|---|---|
| Weekly delayed share of delivered parcels | 0.2317 historically | Outside 0.15 to 0.30 in an ordinary (non-sale) week |
| Weekly share of parcels flagged | 0.25 on the test set (45 of 180) | Below 0.15 or above 0.35 |
| Recall on parcels whose outcome is now known (rolling 4 weeks) | 0.7561 test, 0.8333 validation | Below 0.70: revisit the threshold and retrain |
| Precision on the same parcels | 0.6889 test | Below 0.55: too many unnecessary calls |
| Input drift per hub: weekly mean `distance_km` and `n_stops` against `outputs/hub_profile.csv` | For example, 470.4757 km for the Escalate segment | A move of more than 15% for any hub. Also any unseen `hub_id` or service type: alert and route the parcel for manual review |
| Share of filled values in `distance_km`, `parcel_weight_kg` and `declared_value_inr` | 36 of 1200 distances (0.03) | Above 0.10 in any week, which signals a broken scanner or export |

**(c) Retraining and the sale window**

- **Schedule.** Retrain monthly on a rolling window of delivered parcels, and immediately when an ordinary-week trigger fires. Retraining re-runs Parts A to D unchanged. The new model replaces the current one only if it matches or beats it on the most recent held-out month at the chosen threshold.
- **The 15-28 September sale window** (`sale_event_flag` = 1) raises delays, most of all for Express. `sale_event_flag` is one of the seven selected signals (F-score 36.7281) and correlates +0.1709 with `delivery_hours`.
  - Judge sale weeks against the previous sale window, not against ordinary weeks. A jump in the delayed share or flagged share during the sale is expected, and it must not trigger retraining or a threshold change by itself.
  - Exclude sale weeks from ordinary-week baselines.
  - Every training window must contain at least one past sale event, or the model cannot learn the sale effect. The current history contains only one, the last 14 days of the 90-day window.
- **Data drift** would show up as changes in the inputs:
  - The distance mix shifts toward the 401-600 and 601+ km bands, where delay rates are 0.4364 and 0.8523.
  - Rider routes get longer (`n_stops`).
  - Monsoon weeks bring many more rain days.
  - The Express or COD share changes, or new hubs open.
- **Concept drift** would show up differently. The same inputs would start producing different delays, for example after a hub opens a new sort centre. Recall and precision would fall while the input distributions look stable.

## Limitations

1. **Synthetic data.** Every row comes from a known, mostly additive formula with Gaussian noise. Real parcels have messier relationships that change over time, so R² 0.9313 (validation) and 0.9493 (test) and F1 0.7209 overstate what real data would give.
2. **Few delayed parcels in evaluation.** The test set has 41 delayed parcels and the validation set has 42. One parcel moves recall by about 0.024, so the test recall of 0.7561 is a rough estimate.
3. **Threshold and configuration tuned on one split.** The threshold of 0.5 and the choice of the untuned forest come from a single 180-row validation split. On that split, the three tuning candidates were separated by a few parcels (validation F1 0.775, 0.7848 and 0.8333), so another split could choose differently.
4. **Short history, one season.** The data covers 90 days of dispatches, 1 July to 28 September 2026, with a single sale event. No festive peak, winter fog or year-on-year pattern is represented.
5. **Small hub samples.** Each hub profile rests on 50 parcels, so one parcel changes a hub's delay rate by 0.02. Hubs near a segment boundary could change segment with more data.
6. **Biased importances and a misleading tier feature.** Impurity importance gave 0.2245 to pure noise, so it should not be used to explain the model. Also, `hub_tier` is a model input but is misleading for four hubs.
7. **No explicit cost model.** The brief's cost asymmetry (a missed delay costs more than a call) is encoded as a 0.80 recall floor, not as costs in INR. With real per-call and per-miss costs in INR, the threshold should minimise expected cost.
8. **Pickle portability and trust.** `models/delay_model.pkl` loads reliably only with scikit-learn 1.5.2. Like any pickle, it must be loaded only from a trusted source.
