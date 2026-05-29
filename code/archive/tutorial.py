"""
tutorial.py — interactive walkthrough of the pylater library
=============================================================

Run this script section by section (or all at once) to see:
  1. What the data looks like
  2. How the LATER distribution works in promptness space
  3. How to build and fit a Bayesian model
  4. How to inspect posterior parameters
  5. How to visualise with a reciprobit plot
  6. How to fit multiple conditions with parameter sharing

All values are hard-coded. No files are read or written.


# General model idea:

P(params | data) ∝ P(data | params) × P(params)
     posterior         likelihood        prior

"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pylater


# =============================================================================
# SECTION 1 — What is "promptness"?
# =============================================================================
# LATER works in *promptness* space: λ = 1/RT (units: 1/second).
# A fast response of 100 ms has promptness 10.
# A slow response of 500 ms has promptness 2.
# In promptness space the LATER distribution is Gaussian — that is the key
# insight that makes the maths tractable.

rts_ms = np.array([120, 145, 160, 175, 190, 210, 230, 260, 300, 380])
rts_s  = rts_ms / 1000.0          # convert to seconds — pylater always uses seconds
promptness = 1.0 / rts_s
logging = True


if logging:
    print("=== Section 1: promptness vs RT ===")
    print(f"{'RT (ms)':>10}  {'RT (s)':>8}  {'Promptness (1/s)':>18}")
    for t_ms, t_s, p in zip(rts_ms, rts_s, promptness):
        print(f"{t_ms:>10}  {t_s:>8.3f}  {p:>18.2f}")


# =============================================================================
# SECTION 2 — The Dataset container
# =============================================================================
# pylater.Dataset wraps an array of RTs (in seconds) and pre-computes:
#   .promptness  — 1/rt_s
#   .ecdf        — scipy empirical CDF object
#   .ecdf_p      — ECDF probability values
#   .ecdf_x      — ECDF x-values (RT in seconds)

my_data = pylater.Dataset(name="tutorial", rt_s=rts_s)

if logging:
    print("=== Section 2: Dataset attributes ===")
    print(f"name        : {my_data.name}")
    print(f"rt_s        : {my_data.rt_s}")
    print(f"promptness  : {my_data.promptness.round(2)}")
    print(f"ecdf_p      : {my_data.ecdf_p.round(3)}")   # cumulative probabilities
    print(f"ecdf_x (ms) : {(my_data.ecdf_x * 1000).round(1)}")
    print()


# =============================================================================
# SECTION 3 — Bundled real data (Carpenter & Williams 1995)
# =============================================================================
# The library ships with saccadic eye-movement RTs recorded by Carpenter &
# Williams (1995, Nature).  Participants made eye movements to a target that
# appeared with different prior probabilities.  Keys encode participant (a/b)
# and the percentile label of that condition (p05–p95).
# Higher percentile label → faster responses (higher promptness mean).

cw = pylater.data.cw1995   # dict keyed by "{participant}_p{percentile}"
if logging:
    print("=== Section 3: bundled cw1995 dataset keys ===")
    print(list(cw.keys()))

# Pick one participant, three percentile conditions
d_p05 = cw["a_p05"]   # 5th-percentile condition — slow
d_p50 = cw["a_p50"]   # 50th-percentile condition — medium
d_p95 = cw["a_p95"]   # 95th-percentile condition — fast

if logging:
    print("=== Section 3: dataset summaries ===")
    for d in [d_p05, d_p50, d_p95]:
        print(f"{d.name:8s}  n={len(d.rt_s):3d}  "
            f"median RT={np.median(d.rt_s)*1000:.0f} ms  "
            f"median promptness={np.median(d.promptness):.2f}")



# =============================================================================
# SECTION 4 — The LATER distribution: logp and random samples
# =============================================================================
# pylater.LATER is a PyMC CustomDist.  The underlying functions logp() and
# random() can be called directly for inspection.
#
# Parameters (all in promptness space):
#   mu      — mean of the main (late) Gaussian
#   sigma   — std dev of the main Gaussian
#   sigma_e — std dev of the early Gaussian (mean fixed at 0)
#
# Two independent accumulator processes race to threshold:
#   • Late process  — N(mu, sigma):    the main deliberate decision signal.
#   • Early process — N(0,  sigma_e):  a fast, reflexive process with mean
#                                      zero (equally likely to help or hinder).
#
# A response is triggered by whichever process reaches threshold first, i.e.
# whichever has the higher promptness on a given trial.
#
# The pdf of the winning promptness r is therefore the sum of two terms —
# each "process wins" scenario weighted by the probability that the other
# process has not yet fired:
#
#   p(r) = f_late(r)  * Φ_early(r)   ← late wins,  early still below r
#         + f_early(r) * Φ_late(r)    ← early wins, late  still below r
#
# where f_*(r) is the Gaussian pdf evaluated at r for that process, and
# Φ_*(r) is the corresponding Gaussian CDF (probability the other process
# has promptness ≤ r, i.e. has not yet fired).
#
# The log-probability of a promptness value r is therefore:
#   log[ f_late(r) * Φ_early(r)  +  f_early(r) * Φ_late(r) ]

print("=== Section 4: logp at several promptness values ===")
# Typical parameters for a 200-300 ms RT task
mu_val      = 5.0   # mean promptness ≈ 1/200 ms = 5 /s
sigma_val   = 0.75
sigma_e_val = 3.0   # early component is broader and zero-centred

promptness_grid = np.array([2.0, 4.0, 5.0, 6.0, 8.0, 10.0])
for p in promptness_grid:
    lp = pylater.dist.logp(
        value   = p,
        mu      = mu_val,
        sigma   = sigma_val,
        sigma_e = sigma_e_val,
    ).eval()
    print(f"  logp(promptness={p:5.1f}) = {lp:.4f}")


# Draw samples from the distribution
rng = np.random.default_rng(seed=42)
samples_promptness = pylater.dist.random(
    mu      = mu_val,
    sigma   = sigma_val,
    sigma_e = sigma_e_val,
    rng     = rng,
    size    = 500,
)
samples_rt_ms = 1000.0 / samples_promptness   # convert back to milliseconds

if logging:
    print("=== Section 4: 500 random draws from LATER ===")
    print(f"  mean RT   = {samples_rt_ms.mean():.1f} ms")
    print(f"  median RT = {np.median(samples_rt_ms):.1f} ms")
    print(f"  std  RT   = {samples_rt_ms.std():.1f} ms")
    print(f"  min  RT   = {samples_rt_ms.min():.1f} ms")
    print(f"  max  RT   = {samples_rt_ms.max():.1f} ms")



# =============================================================================
# SECTION 5 — Build a Bayesian model for a single dataset
# =============================================================================
# build_default_model() creates a PyMC model with three free parameters and
# LogNormal priors.  The model hierarchy is:
#
#   sigma        ~ LogNormal(log(0.75), ...)   std dev in promptness space
#   k            ~ LogNormal(log(5),    ...)   mu / sigma  (signal-to-noise)
#   sigma_e_mod  ~ LogNormal(log(4),    ...)   sigma_e / sigma
#
#   mu      = sigma * k            (deterministic)
#   sigma_e = sigma * sigma_e_mod  (deterministic)
#
#   RT | mu, sigma, sigma_e  ~  LATER(mu, sigma, sigma_e)

model_single = pylater.build_default_model(datasets=[d_p50])

if logging:
    print("=== Section 5: single-dataset model structure ===")
    print(model_single.basic_RVs)



# =============================================================================
# SECTION 6 — Prior predictive check
# =============================================================================
# Before fitting, sample from the prior to confirm the priors produce
# plausible RTs (roughly 100–1000 ms for a saccade task).

print("=== Section 6: prior predictive (100 draws) ===")

with model_single:
    idata_single = pm.sample_prior_predictive(samples=100, random_seed=42)

prior_rt_s = idata_single.prior_predictive["obs_a_p50"].values.flatten()
prior_rt_ms = prior_rt_s * 1000.0
if logging:
    print(f"  prior predictive RT:  "
          f"median={np.median(prior_rt_ms):.0f} ms  "
          f"5th={np.percentile(prior_rt_ms,5):.0f} ms  "
      f"95th={np.percentile(prior_rt_ms,95):.0f} ms")

# =============================================================================
# SECTION 7 — MCMC sampling (posterior inference)
# =============================================================================
# pm.sample() runs NUTS MCMC.  The posterior summarises our uncertainty about
# the three parameters given the observed RTs.

print("=== Section 7: MCMC sampling ===")
with model_single:
    idata_single = pm.sample(
        draws          = 500,
        tune           = 500,
        chains         = 2,
        random_seed    = 42,
        idata_kwargs   = {"log_likelihood": True},
        progressbar    = True,
    )

print()
print("Posterior summary:")
summary = az.summary(idata_single, var_names=["sigma", "k", "sigma_e_mod"])
print(summary.to_string())
print()

# Interpret the posterior means
post = idata_single.posterior
sigma_mean    = float(post["sigma"].mean())
k_mean        = float(post["k"].mean())
sigma_e_mean  = float(post["sigma_e_mod"].mean())
mu_mean       = sigma_mean * k_mean
if logging:
    print(f"Derived posterior means:")
    print(f"  mu (mean promptness)   = {mu_mean:.2f}  → median RT ≈ {1000/mu_mean:.0f} ms")
    print(f"  sigma                  = {sigma_mean:.3f}")
    print(f"  sigma_e                = {sigma_mean * sigma_e_mean:.3f}")
    print()


# =============================================================================
# SECTION 8 — Posterior predictive check
# =============================================================================

print("=== Section 8: posterior predictive check ===")
with model_single:
    idata_single = pm.sample_posterior_predictive(
        trace               = idata_single,
        extend_inferencedata= True,
        random_seed         = 42,
    )

post_rt_s  = idata_single.posterior_predictive["obs_a_p50"].values.flatten()
post_rt_ms = post_rt_s * 1000.0
obs_rt_ms  = d_p50.rt_s * 1000.0

if logging:
    print(f"  observed   median RT = {np.median(obs_rt_ms):.0f} ms")
    print(f"  posterior predictive  median RT = {np.median(post_rt_ms):.0f} ms")
    print()


# =============================================================================
# SECTION 9 — Reciprobit plot
# =============================================================================
# A LATER-distributed variable appears as a *straight line* when plotted with:
#   x-axis = promptness (or equivalently 1/RT), reciprocal scale
#   y-axis = cumulative probability, probit (inverse-normal CDF) scale
#
# This is called a reciprobit plot.  Fitting the model is equivalent to fitting
# a straight line on this plot.

print("=== Section 9: reciprobit plot ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel — raw data (ECDF) and posterior model fit
ax1 = axes[0]
rp1 = pylater.ReciprobitPlot(fig_ax=(fig, ax1))
rp1.plot_data(data=d_p50, plot_type="step")
rp1.plot_model(idata=idata_single, dataset_name="a_p50", ci_range=0.95)
ax1.set_title("p=50%: data + posterior model (95% CI)")
# Right panel — posterior predictive
ax2 = axes[1]
rp2 = pylater.ReciprobitPlot(fig_ax=(fig, ax2))
rp2.plot_data(data=d_p50, plot_type="scatter")
rp2.plot_predictive(idata=idata_single, predictive_type="posterior", ci_range=0.95)
ax2.set_title("p=50%: data + posterior predictive (95% CI)")

plt.tight_layout()
plt.savefig("/data/reciprobit_single.png", dpi=120)
print("  saved reciprobit_single.png")
print()


# =============================================================================
# SECTION 10 — Multi-dataset model: SHIFT
# =============================================================================
# The "shift" model shares *sigma* across conditions.  Each condition gets its
# own *k* (i.e. its own mean promptness).  On a reciprobit plot this means the
# fitted lines are *parallel* — same slope, different horizontal position.
#
# This captures the effect of prior probability: higher percentile → higher
# promptness mean → line shifts left (faster responses).

print("=== Section 10: shift model (parallel lines) ===")
model_shift = pylater.build_default_model(
    datasets    = [d_p05, d_p50, d_p95],
    share_type  = "shift",
)

with model_shift:
    idata_shift = pm.sample(
        draws       = 500,
        tune        = 500,
        chains      = 2,
        random_seed = 42,
        progressbar = True,
    )

summary_shift = az.summary(idata_shift, var_names=["sigma", "k"])
print(summary_shift.to_string())
print()


# =============================================================================
# SECTION 11 — Multi-dataset model: SWIVEL
# =============================================================================
# The "swivel" model shares *k* across conditions.  Each condition gets its own
# *sigma*.  On a reciprobit plot the lines pivot around a common point — same
# intercept ratio, different slope.
#
# Swivel captures changes in the *variability* of the decision signal while
# the signal-to-noise ratio (k = mu/sigma) stays constant.

print("=== Section 11: swivel model (pivoting lines) ===")
model_swivel = pylater.build_default_model(
    datasets    = [d_p05, d_p50, d_p95],
    share_type  = "swivel",
)

with model_swivel:
    idata_swivel = pm.sample(
        draws       = 500,
        tune        = 500,
        chains      = 2,
        random_seed = 42,
        progressbar = True,
    )

summary_swivel = az.summary(idata_swivel, var_names=["k", "sigma"])
print(summary_swivel.to_string())
print()


# =============================================================================
# SECTION 12 — Model comparison (shift vs swivel)
# =============================================================================
# Compute LOO-CV for each model, then compare.

print("=== Section 12: model comparison ===")
with model_shift:
    pm.compute_log_likelihood(idata_shift)

with model_swivel:
    pm.compute_log_likelihood(idata_swivel)

idata_shift  = pylater.combine_multiple_likelihoods(
    idata=idata_shift,  combined_var_name="obs", combined_dim_name="trial")

idata_swivel = pylater.combine_multiple_likelihoods(
    idata=idata_swivel, combined_var_name="obs", combined_dim_name="trial")

comparison = az.compare({"shift": idata_shift, "swivel": idata_swivel}, var_name="obs", ic="loo")
if logging:
    print(comparison.to_string())
    print()
    print("Higher elpd_loo = better out-of-sample predictive accuracy.")
    print("A difference > ~4 is considered meaningful.")



# Higher elpd_loo = better out-of-sample predictive accuracy.
# A difference > ~4 is considered meaningful.  help me interpret the results
# Swivel is the better model, and the difference is meaningful. Here's a full breakdown:
# elpd_loo — expected log pointwise predictive density. Higher (less negative) = better.
# - Swivel: -18551.6
# - Shift: -18564.5
# - Swivel wins by 12.9 log-probability units
# elpd_diff — difference relative to the best model (swivel = 0 by definition, shift = 12.9 worse).
# dse — standard error of the difference = 11.6. The difference (~12.9) is just over 1 standard error, so it's meaningful but not overwhelming.
# p_loo — effective number of parameters.
# - Swivel: 6.8, Shift: 5.4 — swivel is slightly more complex, but wins anyway.
# weight — model averaging weights if you were to combine both models.
# - Swivel gets 65%, shift gets 35% — consistent with swivel being better but not decisively so.
# warning=False — no Pareto k diagnostics issues, so the LOO estimates are reliable.
# What this means substantively:
# The swivel model fits better, meaning across the three prior probability conditions (p05, p50, p95) the signal-to-noise ratio k is approximately constant, while sigma (variability) changes between conditions. Prior probability affects how reliably the decision signal is accumulated, not its mean speed. This is a theoretically meaningful result — it suggests the prior manipulates the consistency of the decision process rather than its average rate.