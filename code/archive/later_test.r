# =============================================================================
# later_test.r — per-participant LATER model fitting on Go/No-Go data
# =============================================================================
#
# R translation of sampled_test.py using the LATERmodel package.
# Source: https://unimelbmdap.github.io/LATERmodel/
#
# The LATERmodel package fits the LATER model via maximum-likelihood
# optimisation (frequentist), so there are no posterior samples here.
# The equivalent of pylater's Bayesian posterior summaries (mean ± SD) are
# the MLE point estimates and AIC-based model comparisons available through
# LATERmodel::fit_data() and LATERmodel::compare_fits().
#
# Run section by section (e.g. in RStudio with Ctrl+Enter) or source the
# whole file.  Set DATA_PATH and PLOT_PATH below before running.
#
# Install the package if needed:
#   install.packages("LATERmodel")
# =============================================================================

library(LATERmodel)
library(dplyr)      # for data wrangling (install with: install.packages("dplyr"))
library(ggplot2)    # used internally by LATERmodel; loaded here for transparency

# =============================================================================
# SECTION 0 — LATER model parameter glossary
# =============================================================================
# Before diving into the data, here is what each fitted parameter means in the
# LATER framework (Carpenter 1981; Noorani & Carpenter 2016):
#
#  Parameter  | Model component       | Interpretation
#  -----------+-----------------------+----------------------------------------------
#  mu (= a)   | Primary / cognitive   | Mean promptness (1/RT) of the main LATER
#             |                       | distribution.  Higher mu → faster typical RTs.
#             |                       | This is the "slow, deliberate" cognitive signal
#             |                       | that rises linearly to a threshold.
#  sigma      | Primary / cognitive   | SD of the primary promptness distribution.
#             |                       | Controls how variable the main (cognitive)
#             |                       | response times are.  On a reciprobit plot this
#             |                       | is the reciprocal of the line's slope (s = 1/sigma).
#  k          | Primary / cognitive   | Intercept of the reciprobit line in z-score
#             |                       | space (k = mu / sigma).  Indicates the z-score
#             |                       | at which the line crosses infinite latency.
#             |                       | Shared k → "swivel" model (lines pivot together).
#  sigma_e    | Early / fast-reflex   | SD of the *early* component (mean fixed at 0).
#             |                       | This second distribution captures anomalously
#             |                       | fast, reflexive responses — the "express saccade"
#             |                       | tail.  Only present when with_early_component=TRUE.
#  s          | (derived)             | s = 1 / sigma.  The slope of the reciprobit line.
#
# SHIFT model: datasets share sigma (same slope / same variability in the
#              cognitive component) but differ in mu (location shifts).
# SWIVEL model: datasets share k (same intercept) but differ in sigma
#              (lines rotate around a common point, reflecting changes in the
#               cognitive variability while sharing a common asymptotic limit).
# =============================================================================

cat("LATER model parameter glossary printed above.\n\n")


# =============================================================================
# SECTION 1 — Load data
# =============================================================================

cat("=== Section 1: Load data ===\n")

DATA_PATH <- "/data/simulated_rt_data_go_nogo.csv"

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

cat(sprintf("Loaded %d rows\n", nrow(df)))
cat(sprintf("Groups      : %s\n", paste(sort(unique(df$group)), collapse = ", ")))
cat(sprintf("Participants: %d total\n", length(unique(df$participant_id))))
cat("\n")


# =============================================================================
# SECTION 2 — Per-participant LATER model fitting
# =============================================================================
# For each participant, filter to valid go trials (correct, non-missing RT),
# prepare the data, and fit an individual LATER model via MLE.
#
# pylater used Bayesian sampling (pm.sample) producing a posterior for k and
# sigma.  LATERmodel uses maximum-likelihood optimisation, so we obtain point
# estimates directly.  The k and sigma below are the MLE values equivalent to
# the posterior means in the Python script.

cat("=== Section 2: Per-participant LATER model fitting ===\n")

results <- list()

for (grp in sort(unique(df$group))) {
  group_df <- df[df$group == grp, ]
  participant_ids <- sort(unique(group_df$participant_id))
  cat(sprintf("\n--- Group: %s (%d participants) ---\n",
              grp, length(participant_ids)))

  for (pid in participant_ids) {
    pid_df <- group_df[group_df$participant_id == pid, ]

    # Valid go-trial RTs: correct responses only, no NAs
    valid <- pid_df[
      pid_df$trial_type == "go" &
      pid_df$correct == TRUE &
      !is.na(pid_df$rt_ms),
    ]

    n_valid <- nrow(valid)

    if (n_valid < 5) {
      cat(sprintf("  %s: skipped (only %d valid trials)\n", pid, n_valid))
      next
    }

    # LATERmodel expects a data frame with a 'time' column (ms) and a 'name'
    # column identifying the dataset.
    raw <- data.frame(
      time = valid$rt_ms,
      name = pid
    )

    # prepare_data converts time → promptness (1/RT in seconds) and computes
    # the empirical CDF, which is the format required by fit_data().
    prepared <- LATERmodel::prepare_data(raw_data = raw, time_unit = "ms")

    # Fit a single-dataset LATER model (no sharing; equivalent to
    # pylater.build_default_model with a single Dataset and share_type=None).
    # with_early_component=FALSE matches the pylater default (which does not
    # include sigma_e unless explicitly requested).
    fit <- LATERmodel::fit_data(
      data = prepared,
      with_early_component = FALSE,
      jitter_settings = list(n = 7, seed = 42, processes = 1)
    )

    params <- fit$named_fit_params

    # Extract k and sigma (the two parameters reported in sampled_test.py).
    # mu  → primary component mean promptness    (cognitive LATER signal)
    # sigma → primary component SD of promptness (cognitive variability)
    # k   = mu / sigma, the reciprobit intercept (cognitive signal threshold)
    k_val     <- params[pid, "k"]
    sigma_val <- params[pid, "sigma"]
    mu_val    <- params[pid, "mu"]

    results[[length(results) + 1]] <- list(
      group          = grp,
      participant_id = pid,
      n_valid_trials = n_valid,
      median_rt_ms   = round(median(valid$rt_ms), 1),
      k              = round(k_val,     3),
      sigma          = round(sigma_val, 3),
      mu             = round(mu_val,    3),
      aic            = round(fit$aic,   2)
    )

    cat(sprintf("  %s: n=%2d  mu=%.2f  k=%.2f  sigma=%.3f  AIC=%.1f\n",
                pid, n_valid, mu_val, k_val, sigma_val, fit$aic))
  }
}


# =============================================================================
# SECTION 3 — Group-level summaries across participants
# =============================================================================
# Equivalent to Section 3 of sampled_test.py: pool per-participant MLE
# estimates and compute group means, SDs, and ranges.
#
# Parameter reminder (see Section 0):
#   k     — reciprobit intercept of the *cognitive* LATER component
#   sigma — SD of promptness for the *cognitive* LATER component
#   mu    — mean promptness of the *cognitive* LATER component

cat("\n=== Section 3: Group-level summary of per-participant MLE estimates ===\n\n")

results_df <- do.call(rbind, lapply(results, as.data.frame))
results_df$k     <- as.numeric(results_df$k)
results_df$sigma <- as.numeric(results_df$sigma)
results_df$mu    <- as.numeric(results_df$mu)

for (grp in sort(unique(results_df$group))) {
  g <- results_df[results_df$group == grp, ]
  cat(sprintf("Group: %s  (n=%d participants fitted)\n", grp, nrow(g)))
  cat(sprintf("  k    (cognitive intercept): mean=%.3f  sd=%.3f  range=[%.3f, %.3f]\n",
              mean(g$k), sd(g$k), min(g$k), max(g$k)))
  cat(sprintf("  sigma (cognitive SD)       : mean=%.3f  sd=%.3f  range=[%.3f, %.3f]\n",
              mean(g$sigma), sd(g$sigma), min(g$sigma), max(g$sigma)))
  cat(sprintf("  mu   (cognitive mean)      : mean=%.3f  sd=%.3f  range=[%.3f, %.3f]\n",
              mean(g$mu), sd(g$mu), min(g$mu), max(g$mu)))
  cat("\n")
}

cat("Full results table:\n")
print(results_df, row.names = FALSE)


# =============================================================================
# SECTION 4 — Shift vs Swivel model comparison across groups
# =============================================================================
# Pool all valid go-trial RTs within each group to form one dataset per group,
# then compare:
#   SHIFT  model — shared sigma (same cognitive variability across groups,
#                  differing mean promptness; parallel lines on reciprobit plot)
#   SWIVEL model — shared k     (same cognitive intercept across groups,
#                  differing sigma; lines pivot around a common asymptotic point)
#
# Model selection uses AIC via LATERmodel::compare_fits(), analogous to the
# LOO-CV comparison (az.compare) in sampled_test.py.

cat("\n=== Section 4: Shift vs Swivel model comparison across groups ===\n\n")

# --- 4a: Build one dataset per group ---
group_raw <- data.frame()

for (grp in sort(unique(df$group))) {
  valid <- df[
    df$group == grp &
    df$trial_type == "go" &
    df$correct == TRUE &
    !is.na(df$rt_ms),
  ]
  cat(sprintf("  %s: %d valid go-trial RTs pooled\n", grp, nrow(valid)))
  group_raw <- rbind(
    group_raw,
    data.frame(time = valid$rt_ms, name = grp)
  )
}

cat("\n")

# Prepare the pooled group data
group_data <- LATERmodel::prepare_data(raw_data = group_raw, time_unit = "ms")

# --- 4b: SHIFT model (shared sigma, separate mu) ---
# Shared sigma means all groups have the same spread of the *cognitive*
# component but differ in its mean promptness → parallel lines on reciprobit.
cat("Fitting SHIFT model (shared sigma [cognitive SD], separate mu)...\n")

fit_shift <- LATERmodel::fit_data(
  data            = group_data,
  with_early_component = FALSE,
  share_sigma     = TRUE,
  jitter_settings = list(n = 7, seed = 42, processes = 2)
)

cat("SHIFT model fitted parameters:\n")
print(fit_shift$named_fit_params)

cat(sprintf("\nSHIFT model — sigma (shared, cognitive SD) = %.4f\n",
            fit_shift$named_fit_params[1, "sigma"]))
cat(sprintf("  -> All groups share the same cognitive variability (sigma).\n"))
cat(sprintf("  -> Differences in mu reflect shifts in mean decision speed.\n\n"))

# Check optimiser convergence
if (fit_shift$optim_result$convergence != 0) {
  warning("SHIFT model: optimiser did not converge cleanly. ",
          "Message: ", fit_shift$optim_result$message)
}

# --- 4c: SWIVEL model (shared k, separate sigma) ---
# Shared k means all groups share the same reciprobit intercept (cognitive
# threshold limit), but differ in sigma → lines swivel around that intercept.
cat("Fitting SWIVEL model (shared k [cognitive intercept], separate sigma)...\n")

fit_swivel <- LATERmodel::fit_data(
  data             = group_data,
  with_early_component = FALSE,
  intercept_form   = TRUE,   # tells fit_data that 'a' means k, not mu
  share_a          = TRUE,   # share k across groups
  jitter_settings  = list(n = 7, seed = 42, processes = 2)
)

cat("SWIVEL model fitted parameters:\n")
print(fit_swivel$named_fit_params)

cat(sprintf("\nSWIVEL model — k (shared, cognitive intercept) = %.4f\n",
            fit_swivel$named_fit_params[1, "k"]))
cat(sprintf("  -> All groups share the same reciprobit intercept (k).\n"))
cat(sprintf("  -> Differences in sigma reflect changes in cognitive variability.\n\n"))

if (fit_swivel$optim_result$convergence != 0) {
  warning("SWIVEL model: optimiser did not converge cleanly. ",
          "Message: ", fit_swivel$optim_result$message)
}

# --- 4d: AIC-based model comparison ---
# LATERmodel::compare_fits() ranks models by AIC (lower is better).
# This replaces the LOO-CV (az.compare) step in sampled_test.py.
cat("Model comparison (AIC):\n")

comparison <- LATERmodel::compare_fits(
  fits = list(shift = fit_shift, swivel = fit_swivel)
)

print(comparison)

cat("\nLower AIC = better balance of fit quality and model complexity.\n")
cat("preferred_rel_fit_delta_aic: difference from the preferred model's AIC.\n")
cat("preferred_rel_fit_evidence_ratio: relative likelihood of each model.\n")
cat("A delta_AIC > ~4 is typically considered meaningful evidence.\n\n")

# --- 4e: Annotate what the shared parameter means in each model ---
cat("--- Parameter interpretation for each model ---\n")
cat("SHIFT  (shared sigma):\n")
cat("  sigma = variability of the COGNITIVE / primary LATER component.\n")
cat("  Shared sigma → all groups have identical cognitive RT spread.\n")
cat("  Separate mu  → groups differ in mean cognitive promptness.\n\n")
cat("SWIVEL (shared k):\n")
cat("  k     = intercept of the COGNITIVE / primary LATER component.\n")
cat("  Shared k     → all groups share the same asymptotic RT limit.\n")
cat("  Separate sigma → groups differ in cognitive RT variability.\n\n")


# =============================================================================
# SECTION 5 — Reciprobit plots — all groups in a single panel
# =============================================================================
# Visualise the raw ECDF data and the swivel model fit for all groups.
# LATERmodel::reciprobit_plot() handles the non-linear axes natively.

cat("=== Section 5: Reciprobit plots ===\n")

PLOT_PATH <- "/data/data/reciprobit_groups.png"

# reciprobit_plot returns a ggplot object; save with ggsave.
p <- LATERmodel::reciprobit_plot(
  plot_data  = group_data,
  fit_params = fit_swivel$named_fit_params
) +
  ggplot2::labs(
    title    = "Reciprobit plot — pooled RTs per group (swivel model fit)",
    subtitle = paste0(
      "Swivel model: shared k (cognitive intercept = ",
      round(fit_swivel$named_fit_params[1, "k"], 3),
      "); group lines differ in sigma (cognitive SD)"
    )
  )

ggplot2::ggsave(
  filename = PLOT_PATH,
  plot     = p,
  width    = 8,
  height   = 6,
  dpi      = 150
)

cat(sprintf("  saved %s\n", PLOT_PATH))

# =============================================================================
# SECTION 6 — Final parameter annotation summary
# =============================================================================
# Recap which parameter belongs to which part of the LATER model, printed
# after all analyses so the interpretation is fresh alongside the results.

cat("\n=== Section 6: LATER parameter annotation summary ===\n\n")

cat(
  "  PARAMETER  | COMPONENT              | WHAT IT CAPTURES\n",
  "  -----------+------------------------+------------------------------------------\n",
  "  mu         | PRIMARY (cognitive)    | Mean promptness (1/RT in s^-1).\n",
  "             |                        | Encodes the average rate of the cognitive\n",
  "             |                        | decision signal rising to threshold.\n",
  "             |                        | Higher mu → faster typical responses.\n",
  "  -----------+------------------------+------------------------------------------\n",
  "  sigma      | PRIMARY (cognitive)    | SD of the primary promptness distribution.\n",
  "             |                        | Reflects variability in the cognitive signal.\n",
  "             |                        | On a reciprobit plot: slope = s = 1/sigma.\n",
  "             |                        | SHIFT model: sigma shared across groups.\n",
  "  -----------+------------------------+------------------------------------------\n",
  "  k          | PRIMARY (cognitive)    | Reciprobit intercept = mu / sigma.\n",
  "             |                        | z-score at which latency → infinity.\n",
  "             |                        | SWIVEL model: k shared across groups.\n",
  "  -----------+------------------------+------------------------------------------\n",
  "  sigma_e    | EARLY (fast-reflex)    | SD of the early / express component.\n",
  "             |                        | Mean of this component is fixed at 0.\n",
  "             |                        | Captures anomalously fast reflexive RTs\n",
  "             |                        | (express saccades; not fitted here but\n",
  "             |                        | available via with_early_component=TRUE).\n",
  sep = ""
)

cat("\nDone.\n")
