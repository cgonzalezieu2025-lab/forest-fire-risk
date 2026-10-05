# %% [markdown]
# # EDA: Mesogeos Track A (wildfire danger forecasting)
#
# Data: `data/track_a/{positives,negatives}.csv` from the Mesogeos Google Drive
# (`ml_tracks/a.danger_forecasting`). Each **sample is a 30-day sequence** (30 rows,
# `time_idx` 0..29) for one 1 km grid cell. Positives: day 29 is a fire day (burned
# area from EFFIS). Negatives: random non-fire cell/date, sampled with the same
# seasonal/geographic distribution as positives.
#
# Columns: `time`, `x` (lon), `y` (lat), `sample` (sequence id), `time_idx`,
# 12 dynamic drivers (change daily), 14 static drivers (constant per cell), and
# 3 label/leakage columns (`burned_areas`, `ignition_points`, `burned_area_has`).

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA = ROOT / "data" / "track_a"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

FIRE, NOFIRE = "#eb6834", "#2a78d6"   # categorical slots 2 and 1
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold"})

pos = pd.read_csv(DATA / "positives.csv").assign(label=1)
neg = pd.read_csv(DATA / "negatives.csv").assign(label=0)
# sample ids collide between files -> make them unique
neg["sample"] += pos["sample"].max() + 1
df = pd.concat([pos, neg], ignore_index=True)
df["time"] = pd.to_datetime(df["time"])

vars_dict = json.load(open(DATA / "vars_dict.json"))
LEAK = ["burned_areas", "ignition_points", "burned_area_has"]
DYNAMIC = [v for v in vars_dict["dynamic"].values() if v not in LEAK]
STATIC = [v for v in vars_dict["static"].values() if v not in LEAK]
print(f"rows={len(df):,}  samples={df['sample'].nunique():,}  positives={pos['sample'].nunique():,}  negatives={neg['sample'].nunique():,}")
print("dynamic:", DYNAMIC)
print("static:", STATIC)

# %% [markdown]
# ## 1. Structure checks

# %%
assert (df.groupby("sample").size() == 30).all(), "every sample must have 30 rows"
assert (df.groupby("sample")[STATIC].nunique().max() == 1).all(), "static features must be constant within a sample"
last = df[df.time_idx == 29].set_index("sample")          # one row per sample = the "forecast day"
print("date range:", df.time.min().date(), "->", df.time.max().date())
print("lon:", round(df.x.min(), 2), round(df.x.max(), 2), " lat:", round(df.y.min(), 2), round(df.y.max(), 2))
print(f"class balance (samples): fire={last.label.sum():,}  no-fire={(1-last.label).sum():,}  ratio 1:{(1-last.label).sum()/last.label.sum():.2f}")

# %% [markdown]
# ## 2. Label / leakage columns
# `burned_areas`, `ignition_points` and `burned_area_has` *are* the target. They must
# be dropped from the feature set (the Mesogeos baselines do this too).

# %%
print("positives, day 29: burned_areas>0 =", f"{(last.loc[last.label==1,'burned_areas']>0).mean():.1%}",
      " ignition_points>0 =", f"{(last.loc[last.label==1,'ignition_points']>0).mean():.1%}")
print("positives, days 0-28: any burned_areas>0 =",
      f"{(df[(df.label==1)&(df.time_idx<29)].groupby('sample').burned_areas.max()>0).mean():.1%}")
print("negatives, any day: any burned_areas>0 =",
      f"{(df[df.label==0].groupby('sample').burned_areas.max()>0).mean():.2%}")
print("burned_area_has (ha) for positives:")
print(last.loc[last.label == 1, "burned_area_has"].describe().round(0))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(np.log10(last.loc[last.label == 1, "burned_area_has"]), bins=50, color=FIRE)
ax.set(xlabel="log10 final burned area (ha)", ylabel="fires", title="Fire size is heavy-tailed (median ≈150 ha)")
fig.tight_layout(); fig.savefig(OUT / "01_fire_size.png")

# %% [markdown]
# ## 3. Temporal distribution

# %%
yr = last.groupby([last.time.dt.year, "label"]).size().unstack(fill_value=0)
mo = last.groupby([last.time.dt.month, "label"]).size().unstack(fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2, 1]})
for ax, tab, title in [(axes[0], yr, "Samples per year"), (axes[1], mo, "Samples per month (day 29)")]:
    x = np.arange(len(tab)); w = 0.4
    ax.bar(x - w/2, tab[1], w, color=FIRE, label="fire")
    ax.bar(x + w/2, tab[0], w, color=NOFIRE, label="no fire")
    ax.set_xticks(x); ax.set_xticklabels(tab.index, rotation=45 if len(tab) > 12 else 0); ax.set_title(title)
axes[0].legend(frameon=False); axes[0].axvspan(10.5, 16.5, color="gray", alpha=0.12)
axes[0].text(13.5, yr.values.max()*0.95, "Sentinel-2 overlap\n2017–2022", ha="center", fontsize=9, color="#52514e")
fig.tight_layout(); fig.savefig(OUT / "02_time.png")
print(yr.assign(total=yr.sum(axis=1)).T)
print("2017–2022:", yr.loc[2017:].sum().rename({1: "fire", 0: "no fire"}).to_dict())

# %% [markdown]
# ## 4. Geography

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
n0 = last[last.label == 0]; n1 = last[last.label == 1]
axes[0].scatter(n0.x, n0.y, s=2, color=NOFIRE, alpha=0.25, label="no fire")
axes[0].scatter(n1.x, n1.y, s=2, color=FIRE, alpha=0.5, label="fire")
axes[0].set(title="Sample locations", xlabel="lon", ylabel="lat"); axes[0].legend(frameon=False, markerscale=6)
h = axes[1].hist2d(n1.x, n1.y, bins=[90, 40], cmap="Oranges", cmin=1)
fig.colorbar(h[3], ax=axes[1], label="fires per cell")
axes[1].set(title="Fire density (positives)", xlabel="lon", ylabel="lat")
for ax in axes: ax.set_aspect("equal"); ax.grid(False)
fig.tight_layout(); fig.savefig(OUT / "03_geo.png")

# rough country buckets by lon/lat boxes, for a geographic split later
def region(r):  # ponytail: lon/lat boxes, swap for the Ecoregions/country shapefile if a clean split matters
    if r.y < 36.0 and r.x < 12: return "N. Africa"
    if r.x < -6 and r.y > 36.5: return "Portugal"
    if r.x < 3.5 and r.y > 35.5 and r.y < 44: return "Spain"
    if 3.5 <= r.x < 8 and r.y >= 42: return "France"
    if 6.5 <= r.x < 19 and r.y > 36 and r.y <= 47: return "Italy"
    if 19 <= r.x < 29 and r.y > 34: return "Greece/Balkans"
    if r.x >= 26 and r.y > 35.5: return "Turkey"
    return "other"
last["region"] = last.apply(region, axis=1)
print(last.groupby(["region", "label"]).size().unstack(fill_value=0).rename(columns={1: "fire", 0: "no fire"}).sort_values("fire", ascending=False))

# %% [markdown]
# ## 5. Missing values
# LST (MODIS, cloud-masked) and soil moisture have the most gaps. Negatives have more
# LST gaps than positives (fire days are cloud-free by construction). A model that
# learns "missing LST => no fire" is learning the sampling artifact, so impute
# before training (the baselines use the per-sequence mean).

# %%
na = df.groupby("label")[DYNAMIC + STATIC].apply(lambda g: g.isna().mean()).T
na = na[(na > 0).any(axis=1)].rename(columns={1: "fire", 0: "no fire"}).sort_values("fire", ascending=False)
fig, ax = plt.subplots(figsize=(7, 3))
x = np.arange(len(na)); w = 0.4
ax.bar(x - w/2, na["fire"], w, color=FIRE, label="fire"); ax.bar(x + w/2, na["no fire"], w, color=NOFIRE, label="no fire")
ax.set_xticks(x); ax.set_xticklabels(na.index); ax.set(ylabel="fraction NaN", title="Missing values by class")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "04_missing.png")
print(na.round(3))

# %% [markdown]
# ## 6. Dynamic drivers on the forecast day (fire vs no fire)

# %%
UNITS = {"t2m": "K", "d2m": "K", "rh": "frac", "tp": "m/day", "sp": "Pa", "ssrd": "J/m²", "wind_speed": "m/s",
         "wind_direction": "°", "lst_day": "K", "lst_night": "K", "ndvi": "", "lai": "", "smi": ""}
fig, axes = plt.subplots(4, 4, figsize=(14, 11)); axes = axes.ravel()
for ax in axes[len(DYNAMIC):]: ax.axis("off")
for ax, v in zip(axes, DYNAMIC):
    a, b = last.loc[last.label == 1, v].dropna(), last.loc[last.label == 0, v].dropna()
    lo, hi = np.nanpercentile(pd.concat([a, b]), [0.5, 99.5]); bins = np.linspace(lo, hi, 50)
    ax.hist(b, bins, density=True, color=NOFIRE, alpha=0.6, label="no fire")
    ax.hist(a, bins, density=True, color=FIRE, alpha=0.6, label="fire")
    ax.set_title(f"{v} [{UNITS.get(v, '')}]", fontsize=10); ax.set_yticks([])
axes[0].legend(frameon=False)
fig.suptitle("Day-29 distributions (density)", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "05_dynamic_day29.png")

# standardized mean difference, a quick univariate "how separable" score
smd = ((last[last.label == 1][DYNAMIC + STATIC].mean() - last[last.label == 0][DYNAMIC + STATIC].mean())
       / last[DYNAMIC + STATIC].std()).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(smd.index, smd.values, color=[FIRE if s > 0 else NOFIRE for s in smd.values])
ax.axvline(0, color="#52514e", lw=1); ax.set(xlabel="standardized mean difference (fire − no fire)", title="Which features separate the classes?")
fig.tight_layout(); fig.savefig(OUT / "06_smd.png")
print(smd.round(2))

# %% [markdown]
# ## 7. The 30-day run-up: do drivers build toward the fire day?

# %%
prof = df.groupby(["label", "time_idx"])[DYNAMIC].mean()
fig, axes = plt.subplots(4, 4, figsize=(14, 10.5)); axes = axes.ravel()
for ax in axes[len(DYNAMIC):]: ax.axis("off")
for ax, v in zip(axes, DYNAMIC):
    ax.plot(prof.loc[0, v].index - 29, prof.loc[0, v].values, color=NOFIRE, lw=2, label="no fire")
    ax.plot(prof.loc[1, v].index - 29, prof.loc[1, v].values, color=FIRE, lw=2, label="fire")
    ax.set_title(v, fontsize=10); ax.set_xlabel("days before day 29")
axes[0].legend(frameon=False)
fig.suptitle("Mean driver trajectory over the 30-day window", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "07_trajectories.png")

# %% [markdown]
# ## 8. Static drivers (land cover, terrain, human)

# %%
LC = [s for s in STATIC if s.startswith("lc_")]
OTHER = [s for s in STATIC if s not in LC]
fig, ax0 = plt.subplots(figsize=(7, 3.6)); axes = [ax0]
lc = last.groupby("label")[LC].mean().T.rename(columns={1: "fire", 0: "no fire"})
x = np.arange(len(lc)); w = 0.4
axes[0].bar(x - w/2, lc["fire"], w, color=FIRE, label="fire"); axes[0].bar(x + w/2, lc["no fire"], w, color=NOFIRE, label="no fire")
axes[0].set_xticks(x); axes[0].set_xticklabels([c[3:] for c in lc.index], rotation=35, ha="right")
axes[0].set(title="Mean land-cover fraction", ylabel="fraction"); axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "08_static_landcover.png")

fig, axes = plt.subplots(1, len(OTHER), figsize=(14, 3.4))
for ax, v in zip(axes, OTHER):
    a, b = last.loc[last.label == 1, v].dropna(), last.loc[last.label == 0, v].dropna()
    if v in ("population", "roads_distance"):
        a, b = np.log1p(a), np.log1p(b); v = f"log1p({v})"
    bp = ax.boxplot([b, a], widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "#0b0b0b"})
    for patch, c in zip(bp["boxes"], [NOFIRE, FIRE]): patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["no fire", "fire"]); ax.set_title(v, fontsize=10)
fig.suptitle("Terrain / human drivers", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "08_static_terrain.png")

# %% [markdown]
# ## 9. Feature correlation (day-29 rows)

# %%
corr = last[DYNAMIC + STATIC].corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90); ax.set_yticklabels(corr.columns); ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r"); ax.set_title("Feature correlation")
fig.tight_layout(); fig.savefig(OUT / "09_corr.png")
pairs = corr.where(np.triu(np.ones_like(corr, dtype=bool), 1)).stack()
print("|r| > 0.7:"); print(pairs[pairs.abs() > 0.7].sort_values(key=abs, ascending=False).round(2))

# %% [markdown]
# ## 10. Takeaways
# * **26k samples** (8,574 fire / 17,342 no-fire, ratio 1:2), 2006–2022, whole Mediterranean. Each sample = 30 daily rows × 12 dynamic + 14 static drivers.
# * **Drop `burned_areas`, `ignition_points`, `burned_area_has` from X** – they encode the label.
# * **Sentinel-2 overlap (2017–2022): ~4k fires / ~8k non-fires** – plenty for the CNN branch.
# * Fire days are hotter and drier (`t2m`, `lst_*`, `ssrd` up; `rh`, `smi`, `tp` down). Wind speed barely differs. The gap widens over the last ~10 days of the window, so the sequence carries signal beyond day 29 alone.
# * **Negatives are not season-matched**: they are spread evenly over Mar–Oct while fires peak Jul–Aug. A model can score well just by learning 'hot month'. Consider resampling negatives to the fire month distribution for a fairer test.
# * `slope` values sit in 1.44–1.57 for almost all cells – looks mis-scaled in the extraction; use with care or recompute from `dem`.
# * Fires sit on forest/grassland and steeper slopes; negatives skew to agriculture, shrubland and sparse vegetation. Population and road distance barely differ (negatives were sampled to match).
# * Missingness differs by class (LST) → impute, don't let the model see NaN-ness.
# * Splits: by year (2017–2020 train / 2021–2022 test, matches Mesogeos) or by region column above; never random by row (30 rows per sample are near-duplicates).
