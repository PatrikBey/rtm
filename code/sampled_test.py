"""
sampled_test.py — multi-dataset model comparison on two groups
===============================================================

Load two simulated RT datasets (group1 and group2) from CSV files,
extract valid reaction times from condition 1, build shift and swivel
models, and compare them.

All values are hard-coded. No file checks or error handling.

Run section by section (or all at once).
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pylater


# =============================================================================
# SECTION 1 — Load data from CSV files
# =============================================================================

print("=== Section 1: Load data from CSV files ===")

df_group1 = pd.read_csv("/data/data/simulated_rt_data_single1.csv")
df_group2 = pd.read_csv("/data/data/simulated_rt_data_single2.csv")

print(f"Group 1 shape: {df_group1.shape}")
print(f"Group 2 shape: {df_group2.shape}")
print()


# =============================================================================
# SECTION 2 — Extract valid RTs from condition 1
# =============================================================================
# Condition 1 is "Standard Go/No-Go"
# Filter for:
#   - condition_id == 1
#   - trial_type == "go" (only go trials have valid RTs)
#   - correct == True (exclude errors/omissions)
#   - rt_ms is not NaN

print("=== Section 2: Extract valid RTs from condition 1 ===")

# Group 1
cond1_group1 = df_group1[df_group1["condition_id"] == 1]
valid_group1 = cond1_group1[
    (cond1_group1["trial_type"] == "go") &
    (cond1_group1["correct"] == True) &
    (cond1_group1["rt_ms"].notna())
]
rt_ms_group1 = valid_group1["rt_ms"].values
rt_s_group1 = rt_ms_group1 / 1000.0

# Group 2
cond1_group2 = df_group2[df_group2["condition_id"] == 1]
valid_group2 = cond1_group2[
    (cond1_group2["trial_type"] == "go") &
    (cond1_group2["correct"] == True) &
    (cond1_group2["rt_ms"].notna())
]
rt_ms_group2 = valid_group2["rt_ms"].values
rt_s_group2 = rt_ms_group2 / 1000.0

print(f"Group 1 valid RTs: n={len(rt_ms_group1)}  "
      f"median={np.median(rt_ms_group1):.0f} ms  "
      f"std={np.std(rt_ms_group1):.1f} ms")
print(f"Group 2 valid RTs: n={len(rt_ms_group2)}  "
      f"median={np.median(rt_ms_group2):.0f} ms  "
      f"std={np.std(rt_ms_group2):.1f} ms")
print()


# =============================================================================
# SECTION 3 — Create Dataset objects
# =============================================================================

dataset_group1 = pylater.Dataset(name="group1", rt_s=rt_s_group1)
dataset_group2 = pylater.Dataset(name="group2", rt_s=rt_s_group2)

if True:
    print("=== Section 3: Dataset summaries ===")
    print(f"Group 1:  n={len(dataset_group1.rt_s):3d}  "
          f"median RT={np.median(dataset_group1.rt_s)*1000:.0f} ms  "
          f"median promptness={np.median(dataset_group1.promptness):.2f}")
    print(f"Group 2:  n={len(dataset_group2.rt_s):3d}  "
          f"median RT={np.median(dataset_group2.rt_s)*1000:.0f} ms  "
          f"median promptness={np.median(dataset_group2.promptness):.2f}")
    print()


# =============================================================================
# SECTION 4 — Build SHIFT model (shared sigma, separate k)
# =============================================================================
# Shift model: sigma is shared across groups, each group has its own k
# On reciprobit plot: parallel lines with different horizontal positions

print("=== Section 4: SHIFT model (parallel lines) ===")

model_shift = pylater.build_default_model(
    datasets=[dataset_group1, dataset_group2],
    share_type="shift",
)

with model_shift:
    idata_shift = pm.sample(
        draws=500,
        tune=500,
        chains=2,
        random_seed=42,
        progressbar=True,
    )

summary_shift = az.summary(idata_shift, var_names=["sigma", "k"])
print(summary_shift.to_string())
print()

# Posterior predictive for shift
with model_shift:
    idata_shift = pm.sample_posterior_predictive(
        trace=idata_shift,
        extend_inferencedata=True,
        random_seed=42,
    )

# Compute log likelihood for shift
with model_shift:
    pm.compute_log_likelihood(idata_shift)

print("SHIFT model: posterior predictive and log-likelihood computed.")
print()


# =============================================================================
# SECTION 5 — Build SWIVEL model (shared k, separate sigma)
# =============================================================================
# Swivel model: k is shared across groups, each group has its own sigma
# On reciprobit plot: lines pivot around a common point

print("=== Section 5: SWIVEL model (pivoting lines) ===")

model_swivel = pylater.build_default_model(
    datasets=[dataset_group1, dataset_group2],
    share_type="swivel",
)

with model_swivel:
    idata_swivel = pm.sample(
        draws=500,
        tune=500,
        chains=2,
        random_seed=42,
        progressbar=True,
    )

summary_swivel = az.summary(idata_swivel, var_names=["k", "sigma"])
print(summary_swivel.to_string())
print()

# Posterior predictive for swivel
with model_swivel:
    idata_swivel = pm.sample_posterior_predictive(
        trace=idata_swivel,
        extend_inferencedata=True,
        random_seed=42,
    )

# Compute log likelihood for swivel
with model_swivel:
    pm.compute_log_likelihood(idata_swivel)

print("SWIVEL model: posterior predictive and log-likelihood computed.")
print()


# =============================================================================
# SECTION 6 — Model comparison (SHIFT vs SWIVEL)
# =============================================================================

print("=== Section 6: Model comparison (LOO-CV) ===")

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

if True:
    print(comparison.to_string())
    print()
    print("Higher elpd_loo = better out-of-sample predictive accuracy.")
    print("A difference > ~4 is considered meaningful.")
    print()


# # =============================================================================
# # SECTION 7 — Parameter interpretation
# # =============================================================================

# print("=== Section 7: Parameter interpretation ===")

# # Extract posterior means for shift model
# post_shift = idata_shift.posterior
# sigma_shift = float(post_shift["sigma"].mean())
# k_shift_group1 = float(post_shift["k"].sel(dataset_obs='group1').mean())
# k_shift_group2 = float(post_shift["k"].sel(dataset_obs='group2').mean())
# print("\nSHIFT model (shared σ, separate k):")
# print(f"  Shared sigma = {sigma_shift:.3f}")
# print(f"  k (group 1) = {k_shift_group1:.2f}")
# print(f"  k (group 2) = {k_shift_group2:.2f}")
# print(f"  μ (group 1) = {sigma_shift * k_shift_group1:.2f}")
# print(f"  μ (group 2) = {sigma_shift * k_shift_group2:.2f}")
# print(f"  → Groups differ in mean promptness (shift on reciprobit plot)")

# # Extract posterior means for swivel model
# post_swivel = idata_swivel.posterior
# k_swivel = float(post_swivel["k"].mean())
# sigma_swivel_group1 = float(post_swivel["sigma"].sel(dataset_obs='group1').mean())
# sigma_swivel_group2 = float(post_swivel["sigma"].sel(dataset_obs='group2').mean())
# print("\nSWIVEL model (shared k, separate σ):")
# print(f"  Shared k = {k_swivel:.2f}")
# print(f"  sigma (group 1) = {sigma_swivel_group1:.3f}")
# print(f"  sigma (group 2) = {sigma_swivel_group2:.3f}")
# print(f"  → Groups differ in decision variability (pivot on reciprobit plot)")
# print()


# # =============================================================================
# # SECTION 8 — Reciprobit plots: data from both groups
# # =============================================================================

# print("=== Section 8: Reciprobit plots ===")

# fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# rp = pylater.ReciprobitPlot(fig_ax=(fig, ax))
# rp.plot_data(data=dataset_group1, plot_type="scatter")
# rp.plot_data(data=dataset_group2, plot_type="scatter")
# ax.set_title("Group 1 vs Group 2: Reciprobit plot (condition 1)")
# ax.legend(["Group 1", "Group 2"])

# plt.tight_layout()
# plt.savefig("/data/sampled_reciprobit.png", dpi=120)
# print("  saved sampled_reciprobit.png")
# print()
