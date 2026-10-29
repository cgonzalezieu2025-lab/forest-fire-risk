# Fusion: combining the random forest and the CNN

The last step of the plan in `docs/model_plan.md`: a small **logistic-regression meta-learner**
that combines the random forest's fire probability (tabular drivers) with the CNN's fire
probability (Sentinel-2 tile) into one prediction. Same samples, years and metrics as
`reports/tabular/README.md` and `reports/cnn/README.md`.

> **In one paragraph.** On the 2021-22 test years fusion reaches **AUPRC 0.864** (ROC-AUC 0.917,
> F1 0.780), above the random forest alone (0.829) and the CNN alone (0.749). Season-matched it
> keeps **AUPRC 0.793**, against 0.731 for the random forest: the gain from adding the image
> **doubles** once the "fire season" shortcut is removed (+0.035 → +0.062). A paired bootstrap
> over the test samples puts both gains clearly above zero (95 % intervals [+0.023, +0.046] and
> [+0.042, +0.082]). The meta-learner trusts the random forest more (weight 2.13 against 1.26 per
> standard deviation), but both weights are large, so the image adds information the 27 drivers
> do not have. A plain average of the two probabilities already gets most of the gain (0.846);
> learning the weights adds the rest.

## 1. Files and how to reproduce

| File | What it is |
|---|---|
| `metrics.csv` | Test scores of fusion, the simple average, RF and CNN, `standard` and `season_matched` |
| `meta_learner.json` | Chosen input scale and C, CV AUPRC on 2020, threshold, intercept and the two weights |
| `cv_results.csv` | Every setting tried in the 2020 cross-validation |
| `bootstrap.csv` | Fusion AUPRC minus RF / CNN AUPRC on test: mean and 95 % paired-bootstrap interval |
| `predictions.csv` | `p_rf`, `p_cnn`, `p_fusion` per `sample_id`; for 2020 `p_fusion` is out-of-fold |
| `figures/` | The figures below |

```bash
python -m forest_fire_risk.fusion           # ~10 s, needs only reports/tabular/ and reports/cnn/predictions.csv
python -m forest_fire_risk.fusion_figures   # figures
```

Code: `src/forest_fire_risk/fusion.py`, `fusion_figures.py`; tests in `tests/test_fusion.py`.
Interactive companion: `notebooks/fusion.ipynb` (source `notebooks/fusion.py`, jupytext); every
cell runs without the raw data, tiles or GPU.
No raw data, tiles or GPU are needed: fusion only reads the two committed prediction files.

## 2. Data and years

- **Inputs:** `p_rf` from `reports/tabular/predictions.csv` and `p_cnn` from
  `reports/cnn/predictions.csv`, joined on `sample_id`. The loader checks that both files hold
  exactly the same samples with the same label and split (2,098 in 2020, 3,758 in 2021-22).
- **Why 2020 (stacking without leakage):** neither base model was fitted or tuned on 2020, so
  their 2020 probabilities are as honest as their test probabilities. If the meta-learner were
  trained on predictions for samples the base models had already seen, those predictions would
  look better than they really are and the meta-learner would trust them too much.

| Years | Role here |
|---|---|
| 2017-2019 | not used (the base models were trained on it) |
| 2020 | fit the meta-learner; 5-fold stratified CV picks its settings and threshold |
| 2021-2022 | test, scored once at the end |

## 3. The model, step by step

For each sample the base models give two probabilities, $p_{rf}$ and $p_{cnn}$.

1. **Logit (optional).** $\text{logit}(p) = \log\frac{p}{1-p}$ maps $(0,1)$ to $(-\infty,\infty)$.
   The CNN overfits and outputs very extreme probabilities (as low as $2\cdot10^{-5}$), so on the
   probability scale its strong opinions get squashed against 0 and 1. Probabilities are clipped
   to $[10^{-6}, 1-10^{-6}]$ first.
2. **Standardise** each input with the mean and standard deviation from 2020: $z = (v - \mu)/\sigma$.
3. **Logistic regression with an L2 penalty:**
   $$p_{final} = \sigma(w_0 + w_{rf}\, z_{rf} + w_{cnn}\, z_{cnn}), \qquad \sigma(t) = \frac{1}{1+e^{-t}}$$
   Because the inputs are standardised, $w_{rf}$ and $w_{cnn}$ are both "change in log-odds of
   fire per standard deviation of that model's output" and can be compared directly.
4. **Tuning.** 5-fold stratified cross-validation inside 2020, scored by AUPRC, over the input
   scale (probability or logit) and $C \in \{0.001, \dots, 100\}$ ($C = 1/\lambda$). Then refit on all of 2020.
5. **Threshold.** The F1-maximising cut-off on the out-of-fold 2020 predictions (same rule as the
   tabular models). The test set is not used for any choice.

**What was chosen.** Logit inputs win in every setting (CV AUPRC 0.891 against 0.884 for raw
probabilities). $C$ barely matters from 0.1 upwards (scores equal to three decimals), which is
expected with only two inputs and 2,098 samples. Threshold 0.38. Re-running with four different
CV seeds gives the same test AUPRC (0.864) every time.

$$p_{final} = \sigma(-1.35 + 2.13\, z_{rf} + 1.26\, z_{cnn})$$

**Two comparison points:** the random forest and the CNN alone (with the thresholds from their
own reports, picked on 2017-19), and the **simple average** $(p_{rf} + p_{cnn})/2$: a fusion
that learns nothing, with its threshold picked on 2020. It shows how much of the gain comes
from combining at all and how much from learning the weights.

![Scatter of the two base models with the fusion boundary](figures/01_scatter_boundary.png)

**Figure 1.** Every test sample at (RF logit, CNN logit), coloured by the true label. Fires
(orange) sit top-right, non-fires (blue) bottom-left, but each axis alone leaves a wide overlap.
The purple line is where fusion's probability equals its threshold. It is tilted: a sample the
random forest is unsure about (logit around 0) gets classified by the image, and a very
confident CNN can outvote a lukewarm random forest. A vertical line would be "RF only".

## 4. Results on the test years 2021-22

| Model | AUPRC | ROC-AUC | F1 | Precision | Recall | AUPRC season-matched | ROC-AUC season-matched | F1 season-matched |
|---|---|---|---|---|---|---|---|---|
| **Fusion (RF + CNN)** | **0.864** | **0.917** | **0.780** | **0.726** | 0.842 | **0.793** | **0.864** | **0.697** |
| Average of RF and CNN | 0.846 | 0.901 | 0.750 | 0.675 | 0.842 | 0.786 | 0.852 | 0.687 |
| Random forest | 0.829 | 0.899 | 0.758 | 0.691 | 0.839 | 0.731 | 0.831 | 0.671 |
| CNN | 0.749 | 0.843 | 0.695 | 0.656 | 0.738 | 0.705 | 0.810 | 0.657 |

The test set has 3,758 samples (1,319 fires); random guessing gives AUPRC 0.351.

![ROC and precision-recall curves](figures/02_roc_pr_curves.png)

**Figure 2.** ROC (left) and precision-recall (right) curves on the test set. Fusion is on top
almost everywhere. At the same recall as the random forest (84 %), fusion's alarms are real
73 % of the time against 69 %.

![AUPRC bars](figures/03_auprc_bars.png)

**Figure 3.** Test AUPRC, plain (dark) and season-matched (light), with the bootstrap interval
for fusion minus the random forest.

**Is the gain real?** A paired bootstrap resamples the 3,758 test samples with replacement
2,000 times and recomputes "fusion AUPRC minus base-model AUPRC" on each resample (season
weights recomputed each time):

| Fusion minus | Standard: mean [95 % interval] | Season-matched: mean [95 % interval] |
|---|---|---|
| Random forest | +0.035 [+0.023, +0.046] | +0.062 [+0.042, +0.082] |
| CNN | +0.114 [+0.097, +0.132] | +0.087 [+0.064, +0.113] |

Fusion is ahead in 100 % of the resamples in all four cases.

**Reading the numbers.**
- **The two branches are complementary.** Their predictions correlate only moderately
  (0.63 on 2020, 0.59 on test), so they often disagree, and the meta-learner gives both a large
  weight. Neither model is redundant.
- **The image matters more once the season shortcut is gone.** Season matching costs the random
  forest 0.098 AUPRC but fusion only 0.071, and the gain over the random forest grows from +0.035
  to +0.062. This matches the CNN report: the tabular drivers partly encode "what month is it"
  (temperature, radiation), while the tile mostly encodes "what kind of place is this" (fuel,
  vegetation, terrain), which is the part of fire danger the season does not explain.
- **The random forest still carries most of the weight** (2.13 vs 1.26), as expected: it is the
  stronger single model and it sees the weather of the day, which the image cannot.
- **The image works mainly as a veto on false alarms.** At their own thresholds, fusion catches
  about as many test fires as the random forest (1,111 vs 1,106) but raises 75 fewer false alarms
  (419 vs 494). Of the 215 samples the random forest flags and fusion clears, 70 % are correctly
  cleared; the 145 opposite switches are right about half the time (`notebooks/fusion.ipynb`, section 6).
- **Learning the weights is worth it, but most of the gain is in combining at all.** The plain
  average gets 0.846, about half-way from the random forest to fusion; the learned weights and
  the logit scale add another 0.018.

## 5. Limitations

- **The bootstrap covers test-sample noise, not training noise.** The CNN report measured
  ±0.02-0.03 AUPRC between two CNN trainings. The fusion gain over the random forest (+0.035,
  season-matched +0.062) is larger than that, but a re-trained CNN would give somewhat different
  fusion numbers. Re-running the CNN a few times and fusing each run would give the honest spread.
- **One year to fit the meta-learner.** 2020 is a single fire season (2,098 samples); the
  meta-learner has only three parameters, so this is enough to fit it, but its weights reflect
  how the base models behaved in that one year.
- **2020 looks easier than 2021-22** (CV AUPRC 0.891 on 2020 vs 0.864 on test), so the
  threshold picked on 2020 may not be the best one for later years. AUPRC and ROC-AUC do not
  depend on the threshold; F1, precision and recall do.
- **Late fusion only.** The meta-learner sees two numbers, not the features or pixels. A model
  that learns from both inputs jointly (e.g. CNN features concatenated with the tabular
  features) could find interactions this cannot, at the cost of the clean "how much does each
  branch count" reading.
- **Same caveats as the base models:** same-day danger rather than a forecast, the same cells
  can appear in train and test years, season bias handled at evaluation time only.
