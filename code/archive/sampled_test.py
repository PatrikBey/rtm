"""
sampled_test.py — per-participant LATER model fitting on Go/No-Go data
======================================================================

Load simulated RT data from simulated_rt_data_go_nogo.csv, then for every
participant in every group extract valid go-trial RTs, build a single-dataset
LATER model, sample from it, and collect posterior summaries.

Intended to run inside the rtm Docker container:
    docker run --rm -it \\
        -v $(pwd)/code:/workspace/code \\
        -v /mnt/h/RT/data:/data \\
        rtm:dev python code/sampled_test.py

Run section by section (or all at once).
"""

import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pylater

# Suppress pylater default-prior warnings — we are aware they are defaults
warnings.filterwarnings("ignore", category=UserWarning, module="pylater")

# =============================================================================
# SECTION 1 — Load data
# =============================================================================

print("=== Section 1: Load data ===")

DATA_PATH = "/data/simulated_rt_data_go_nogo.csv"

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} rows")
print(f"Groups      : {sorted(df['group'].unique())}")
print(f"Participants: {df['participant_id'].nunique()} total")
print()


# =============================================================================
# SECTION 2 — Per-participant model fitting
# =============================================================================

print("=== Section 2: Per-participant LATER model fitting ===")

results = []

for group in sorted(df["group"].unique()):
    group_df = df[df["group"] == group]
    participant_ids = sorted(group_df["participant_id"].unique())
    print(f"\n--- Group: {group} ({len(participant_ids)} participants) ---")
    for pid in participant_ids:
        pid_df = group_df[group_df["participant_id"] == pid]
        # Valid go-trial RTs: correct responses only, no NaNs
        valid = pid_df[
            (pid_df["trial_type"] == "go") &
            (pid_df["correct"] == True) &
            (pid_df["rt_ms"].notna())
        ]
        rt_s = valid["rt_ms"].values / 1000.0
        if len(rt_s) < 5:
            print(f"  {pid}: skipped (only {len(rt_s)} valid trials)")
            continue
        dataset = pylater.Dataset(name=pid, rt_s=rt_s)
        # share_type=None is the correct choice for a single dataset
        model = pylater.build_default_model(datasets=[dataset])
        with model:
            idata = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                random_seed=42,
                progressbar=False,
            )
        post = idata.posterior
        k_mean     = float(post["k"].mean())
        k_sd       = float(post["k"].std())
        sigma_mean = float(post["sigma"].mean())
        sigma_sd   = float(post["sigma"].std())
        results.append({
            "group":           group,
            "participant_id":  pid,
            "n_valid_trials":  len(rt_s),
            "median_rt_ms":    round(np.median(rt_s) * 1000, 1),
            "k_mean":          round(k_mean,     3),
            "k_sd":            round(k_sd,       3),
            "sigma_mean":      round(sigma_mean, 3),
            "sigma_sd":        round(sigma_sd,   3),
        })
        print(f"  {pid}: n={len(rt_s):2d}  "
              f"k={k_mean:.2f}±{k_sd:.2f}  "
              f"sigma={sigma_mean:.3f}±{sigma_sd:.3f}")


# =============================================================================
# SECTION 3 — Group-level summaries across participants
# =============================================================================

print("\n=== Section 3: Group-level summary of per-participant posterior means ===\n")

results_df = pd.DataFrame(results)

for group in sorted(results_df["group"].unique()):
    g = results_df[results_df["group"] == group]
    print(f"Group: {group}  (n={len(g)} participants fitted)")
    print(f"  k     : mean={g['k_mean'].mean():.3f}  "
          f"sd={g['k_mean'].std():.3f}  "
          f"range=[{g['k_mean'].min():.3f}, {g['k_mean'].max():.3f}]")
    print(f"  sigma : mean={g['sigma_mean'].mean():.3f}  "
          f"sd={g['sigma_mean'].std():.3f}  "
          f"range=[{g['sigma_mean'].min():.3f}, {g['sigma_mean'].max():.3f}]")
    print()

print("Full results table:")
print(results_df.to_string(index=False))


# =============================================================================
# SECTION 4 — Shift vs Swivel model comparison across groups
# =============================================================================
# One Dataset per group, pooling all valid go-trial RTs from every participant
# in that group. The shift model shares sigma across groups (parallel lines on
# the reciprobit plot); the swivel model shares k (lines pivot around a common
# point). LOO-CV is used to compare predictive accuracy.

print("\n=== Section 4: Shift vs Swivel model comparison across groups ===\n")

# --- 4a: Build one Dataset per group ---
group_datasets = []
for group in sorted(df["group"].unique()):
    valid = df[
        (df["group"] == group) &
        (df["trial_type"] == "go") &
        (df["correct"] == True) &
        (df["rt_ms"].notna())
    ]
    rt_s = valid["rt_ms"].values / 1000.0
    group_datasets.append(pylater.Dataset(name=group, rt_s=rt_s))
    print(f"  {group}: {len(rt_s)} valid go-trial RTs pooled")

print()

# --- 4b: Shift model (shared sigma, separate k) ---
print("Fitting SHIFT model (shared sigma, separate k)...")

model_shift = pylater.build_default_model(
    datasets=group_datasets,
    share_type="shift",
)

with model_shift:
    idata_shift = pm.sample(
        draws=500,
        tune=500,
        chains=2,
        random_seed=42,
        progressbar=False,
    )

print(az.summary(idata_shift, var_names=["sigma", "k"]).to_string())
print()

with model_shift:
    pm.sample_posterior_predictive(
        trace=idata_shift, extend_inferencedata=True, random_seed=42
    )
    pm.compute_log_likelihood(idata_shift)

# --- 4c: Swivel model (shared k, separate sigma) ---
print("Fitting SWIVEL model (shared k, separate sigma)...")

model_swivel = pylater.build_default_model(
    datasets=group_datasets,
    share_type="swivel",
)

with model_swivel:
    idata_swivel = pm.sample(
        draws=500,
        tune=500,
        chains=2,
        random_seed=42,
        progressbar=False,
    )

print(az.summary(idata_swivel, var_names=["k", "sigma"]).to_string())
print()

with model_swivel:
    pm.sample_posterior_predictive(
        trace=idata_swivel, extend_inferencedata=True, random_seed=42
    )
    pm.compute_log_likelihood(idata_swivel)

# --- 4d: LOO-CV model comparison ---
print("Model comparison (LOO-CV):")

idata_shift_combined = pylater.combine_multiple_likelihoods(
    idata=idata_shift, combined_var_name="obs", combined_dim_name="trial"
)
idata_swivel_combined = pylater.combine_multiple_likelihoods(
    idata=idata_swivel, combined_var_name="obs", combined_dim_name="trial"
)

comparison = az.compare(
    {"shift": idata_shift_combined, "swivel": idata_swivel_combined},
    var_name="obs",
    ic="loo",
)

print(comparison.to_string())
print()
print("Higher elpd_loo = better out-of-sample predictive accuracy.")
print("A difference > ~4 is considered meaningful.")


# =============================================================================
# SECTION 5 — Reciprobit plot — all groups in a single panel
# =============================================================================
# All three groups are overlaid on one reciprobit plot. Each group has its own
# color applied consistently to the ECDF scatter and the posterior fit band.

print("\n=== Section 5: Reciprobit plots ===")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside the container
import matplotlib.pyplot as plt

PLOT_PATH = "/data/data/reciprobit_groups.png"

colors = {"control": "#2196F3", "frontal": "#F44336", "non-frontal": "#4CAF50"}

fig, ax = plt.subplots(figsize=(8, 6))
rp = pylater.ReciprobitPlot(fig_ax=(fig, ax))

for dataset in group_datasets:
    color = colors[dataset.name]
    rp.plot_data(
        data=dataset,
        plot_type="scatter",
        color=color,
        label=dataset.name,
        zorder=3,
    )
    rp.plot_model(
        idata=idata_swivel,
        dataset_name=dataset.name,
        fill_kwargs={"alpha": 0.15, "color": color},
        line_kwargs={"color": color},
    )

# Build legend:
#   Section 1 — one entry per group (color key)
#   Section 2 — one entry per plot element (marker / line / ribbon)
from matplotlib.patches import Patch
import matplotlib.lines as mlines

group_handles = [
    mlines.Line2D([0], [0], color=colors[ds.name], linewidth=2,
                  marker="o", markersize=5, label=ds.name)
    for ds in group_datasets
]

style_handles = [
    mlines.Line2D([0], [0], color="gray", linestyle="none",
                  marker="o", markersize=5,  label="ECDF data"),
    mlines.Line2D([0], [0], color="gray", linewidth=2,
                  label="posterior median"),
    Patch(facecolor="gray", alpha=0.30,    label="95% credible interval"),
]

# Blank spacer entry used as a visual section divider
spacer = mlines.Line2D([], [], linestyle="none", label="")

all_handles = group_handles + [spacer] + style_handles
ax.legend(handles=all_handles, fontsize=9, framealpha=0.9)
ax.set_title("Reciprobit plot — pooled RTs per group (swivel model fit)")

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"  saved {PLOT_PATH}")
