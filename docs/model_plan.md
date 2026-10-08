# Model plan

Our modeling strategy for the forest-fire-risk project, with the reasoning behind each
decision. Three branches — tabular, image, fusion — building from a simple linear
baseline up to a combined multi-modal model.

## Overview

```
TABULAR branch   features --> [impute, standardize] --> Logistic Regression   (baseline, reported)
                 features --> [impute]                --> Random Forest        (best tabular, base model)

IMAGE branch     Sentinel-2 tiles                     --> CNN                   (base model)

FUSION           [RF prob, CNN prob] --> Logistic Regression (meta-learner)     --> final prediction
```

Everything is evaluated on the **Mesogeos year split** (train <=2019 / val 2020 /
test 2021-22) via `split_by_year`, so all branches are judged on the same samples.
We never split randomly by row: the 30 rows of a sample are near-duplicates and would
leak between train and test.

---

## Step 1 - Logistic Regression (tabular baseline)

**What:** Linear model on the `to_tabular` feature matrix. Preprocessing: impute NaNs,
standardize features, apply **L2 regularization**.

**Why:**
- It is the **reference floor**. Every later model (RF, CNN, fusion) is judged by how
  much it beats this. Without it we have no yardstick.
- **Interpretable** — coefficients give direction and strength per driver, pairing
  naturally with the SMD plot in the EDA (`reports/eda/06_smd.png`). Good for the report.
- Fast to train and reason about.

**Why L2 (and not PCA) for collinearity:** The EDA shows strong multicollinearity
(temperature family `t2m`/`lst_*`/`d2m`, `lai`/`ndvi`, `sp`/`dem`). This destabilizes
LR coefficients. L2 (ridge) stabilizes them **without** sacrificing interpretability.

**Caveat we expect to see:** The EDA predicts LR will underperform — fire risk is
interaction-heavy ("hot AND dry AND forest AND steep"), which a linear model can't
capture without hand-engineered product terms. That underperformance is itself a
finding: it motivates the nonlinear model.

---

## Step 2 - Random Forest (tabular, nonlinear)

**What:** Tree ensemble on the **same** features and the **same** year split.
Preprocessing: impute NaNs only (no scaling needed).

**Why we expect it to win on this data:**
- **Captures interactions automatically** — the "hot AND dry AND forest AND steep"
  structure that drives fire risk, which LR misses.
- **Immune to multicollinearity** — the temperature-family correlations that
  destabilize LR are a non-issue for tree splits.
- **Scale-invariant** — handles mixed types/scales (land-cover fractions, Kelvin
  temps, population, road distance) with no standardization, and shrugs off the
  **mis-scaled `slope`** feature the EDA flagged (stuck in a 1.44-1.57 band).
- **Feature importances** give a second, often more trustworthy view than LR
  coefficients under collinearity — useful for the write-up.
- The Mesogeos reference numbers (sequence models at F1 ~0.79) suggest nonlinearity
  pays; tree ensembles are the usual strong tabular performer.

**Preprocessing is per-model, not global:** LR needs impute + standardize; RF needs
impute only. There is no single shared preprocessing step.

---

## Step 3 - Compare the two tabular models

Evaluate both **identically** on the year split and report:
- **F1**, **AUPRC**, **ROC-AUC**.
- **AUPRC emphasized** because of the 1:2 class imbalance (8,574 fire / 17,342 no-fire).

**The deliverable is the comparison itself** — "nonlinear models buy us X over the
linear baseline" — which mirrors the linear-vs-nonlinear question directly and is a
stronger story than either model alone. If forced to ship one, it would be RF.

---

## PCA - not in the main workflow

**We do not use PCA.** With only ~27 features there is nothing to compress, and PCA
would destroy the interpretability we rely on (LR coefficients, RF importances).
Collinearity is handled by L2 for LR and is a non-issue for RF. PCA may be run as an
optional side experiment or for a 2D visualization only — its components are never fed
to the models.

---

## Step 4 - CNN (image branch)

**What:** CNN on Sentinel-2 RGB tiles pulled from Google Earth Engine for the
2017-2022 samples (~3,968 fires / 8,025 non-fires), same year split.

**Why a CNN:** Spatial structure in imagery (fuel patterns, terrain, burn scars) is
exactly what convolutions are built for; a tabular model can't use raw pixels.

---

## Step 5 - Fusion (meta-learner)

The tabular and image branches each output a **single fire probability** per sample.
Fusion learns how to combine those two probabilities into one final prediction, using a
small **Logistic Regression meta-learner**.

### How it works

For a given sample, each base model produces a probability:

- `p_rf`  = Random Forest's fire probability (from tabular drivers)
- `p_cnn` = CNN's fire probability (from the Sentinel-2 image)

The meta-learner combines them:

```
z            = w0 + w_rf * p_rf + w_cnn * p_cnn
p_final      = sigmoid(z)          # final fire probability in [0, 1]
```

```
  tabular drivers --> Random Forest --> p_rf  --\
                                                  >--> LR meta-learner --> p_final
  Sentinel-2 tile --> CNN           --> p_cnn --/
```

The meta-learner learns the weights `w_rf` and `w_cnn` — i.e. **how much to trust each
model**. Because the inputs are just two probabilities, the weights are directly
readable (e.g. "image evidence weighs more than tabular, or vice versa").

### Why Logistic Regression for the meta-learner

- The input is **only two numbers** (`p_rf`, `p_cnn`), so the combination is a simple,
  near-linear problem — LR fits it well and a more complex combiner would just overfit.
- LR gives **interpretable weights** on each model, which is exactly the "how much does
  each branch contribute" answer we want.

Note this is a **different role** from Step 1: there LR is a weak baseline over ~27 raw
features; here it is combining two probabilities, a task it suits perfectly. The **Step 1
baseline LR is NOT part of the meta-learner** — only RF and the CNN feed it. We leave the
baseline LR out because it uses the same tabular features as RF and would be redundant;
RF and the CNN are different views of the sample, which is what makes combining them worthwhile.

### Anti-leakage rule

The meta-learner must be trained on base-model predictions made on data the base models
did **not** train on (out-of-fold predictions, or a dedicated held-out layer). Training it
on predictions over the base models' own training data silently inflates the scores.

---

## Cross-cutting caveats (from the EDA)

- **Negatives are not season-matched** — spread evenly Mar-Oct while fires peak Jul-Aug,
  so any model can score partly by learning "hot month." Evaluate with and without the
  seasonal resampling of negatives to know how much of the score is real. Applies to
  every model.
- **Imputation must be consistent** across models so comparisons are fair. LST (MODIS)
  gaps differ by class (fire days are cloud-free by construction) — impute so the model
  doesn't learn the missingness pattern itself.
- **Leakage columns** `burned_areas`, `ignition_points`, `burned_area_has` are the label
  in disguise — never features. The `DYNAMIC`/`STATIC` lists in `data.py` already exclude
  them.

---

## Build order

1. LR baseline (tabular).
2. RF (tabular), compare on F1 / AUPRC / ROC-AUC — confirm RF wins before investing further.
3. GEE Sentinel-2 puller + CNN (image).
4. LR meta-learner combining RF + CNN probabilities (fusion), with out-of-fold predictions.
