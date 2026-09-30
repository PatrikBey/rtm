#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                GRAPH REPRESENTATION OF INTELLIGENCE                   #
#                                                                       #
# The following script contains utility functions for single-parameter  #
# reaction-time modelling, used to derive one robust per-patient        #
# descriptive score for downstream SBM modelling.                       #
#                                                                       #
# Two model families are provided:                                      #
#                                                                       #
#   (A) Two-parameter shifted distributions fitted per subject by ML    #
#       -- shifted Wald (inverse Gaussian) and shifted lognormal. Both  #
#       carry intrinsic right skew, so the tail is captured structur-   #
#       ally rather than by a separate third parameter, which is far    #
#       better identified than a free ex-Gaussian tau at ~18 trials.    #
#                                                                       #
#   (B) A hierarchical ex-Gaussian with an explicit lapse component,    #
#       built as a PyMC model. Trials are modelled as a mixture of an   #
#       ex-Gaussian and a wide Uniform "lapse" density, so contaminated #
#       trials are absorbed by the lapse component instead of inflating #
#       tau; log tau is given a group-level prior so per-subject        #
#       estimates are shrunk away from both the tau->0 boundary and     #
#       the outlier-driven blow-up. This is what keeps the right tail   #
#       -- the effect of interest -- in the model rather than trimming  #
#       it away.                                                        #
#                                                                       #
# NOTE on the RT scale: the raw trial file records values ~10x larger   #
# than plausible RTs in ms (control median 3192, within-subject SD 695, #
# foreperiod effect 272). Dividing by 10 puts all four of those in      #
# normative range (319 ms / 70 ms / 27 ms, 96% of trials in            #
# 150-1000 ms). scale_factor therefore defaults to 10, but this is an   #
# inference from distributional plausibility, NOT from the recording    #
# software's documentation -- confirm before relying on absolute        #
# values. Note the scale is irrelevant to any rank-based downstream     #
# analysis: it divides every fitted location/scale parameter by the     #
# same constant.                                                        #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


import csv
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr

#########################################
#                                       #
#         LOGGING UTILITIES             #
#                                       #
#########################################


def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, os, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')


#########################################
#                                       #
#             I/O UTILITIES             #
#                                       #
#########################################


def load_rt_trials(file_path, id_col=0, label_row=1, first_data_row=2, first_trial_col=7,
                   scale_factor=10.0):
    '''
    load the wide per-trial RT table

    layout: row 0 is a title/header row, row `label_row` carries the per-trial
    condition label (e.g. 'long'/'short') from column `first_trial_col` onward,
    and rows from `first_data_row` are one subject each with the subject id in
    column `id_col`. Non-numeric / empty trial cells are dropped per subject.

    returns (subjects, trial_labels, rt_by_subject) where rt_by_subject maps
    subject -> {condition_label: array of RTs divided by scale_factor}
    '''
    with open(file_path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))
    trial_labels = [l.strip() for l in rows[label_row][first_trial_col:]]
    conditions = sorted(set(l for l in trial_labels if l))
    subjects = []
    rt_by_subject = {}
    for row in rows[first_data_row:]:
        subject = row[id_col].strip()
        if not subject:
            continue
        values = []
        for cell in row[first_trial_col:]:
            try:
                values.append(float(cell.strip()))
            except (ValueError, AttributeError):
                values.append(np.nan)
        values = np.asarray(values[:len(trial_labels)], dtype=np.float64)
        if values.size < len(trial_labels):
            values = np.concatenate([values, np.full(len(trial_labels) - values.size, np.nan)])
        per_condition = {}
        for cond in conditions:
            mask = np.array([l == cond for l in trial_labels])
            vals = values[mask]
            per_condition[cond] = vals[np.isfinite(vals)] / float(scale_factor)
        subjects.append(subject)
        rt_by_subject[subject] = per_condition
    return subjects, trial_labels, rt_by_subject


def load_rt_trials_sectioned(file_path, id_col=0, section_row=0, label_row=1,
                             first_data_row=2, section_marker='Speed', scale_factor=10.0):
    """
    load a per-trial RT table laid out in labelled SECTIONS rather than one
    flat run of trials (the Go/No-Go export).

    That file holds eight blocks of 24 columns: four accuracy blocks, one per
    condition, followed by four reaction-time blocks for the same conditions.
    Only the first RT block carries the marker in its header ("Speed:
    A_Condition"); the rest are labelled just "E_Condition", "Up_Condition",
    "Down_Condition" -- names that ALSO appear in the accuracy half. Matching
    on the marker alone therefore picks up only the first block, so the RT
    section is taken positionally: from the first column whose section header
    starts with `section_marker` through to the end of the row.

    Within that span the section header is forward-filled to give each column
    its condition, with the "marker: " prefix stripped, so conditions come out
    as A_Condition / E_Condition / Up_Condition / Down_Condition. Blank cells
    are no-go trials or misses and are simply absent from that subject's data,
    which is why subjects have far fewer RTs than columns.

    Returns the same (subjects, trial_labels, rt_by_subject) triple as
    load_rt_trials, so the rest of the pipeline is unchanged.
    """
    with open(file_path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))

    sections = rows[section_row]
    start = next((i for i, s in enumerate(sections) if s.strip().startswith(section_marker)), None)
    if start is None:
        raise ValueError(f"no section header starting with {section_marker!r} in row {section_row}")

    filled, current = [], ''
    for s in sections:
        if s.strip():
            current = s.strip()
            if current.startswith(section_marker):
                current = current.split(':', 1)[-1].strip()
        filled.append(current)

    cols = list(range(start, len(sections)))
    trial_labels = [filled[i] for i in cols]
    conditions = sorted(set(trial_labels))

    subjects, rt_by_subject = [], {}
    for row in rows[first_data_row:]:
        subject = row[id_col].strip()
        if not subject:
            continue
        values = []
        for i in cols:
            cell = row[i].strip() if i < len(row) else ''
            try:
                values.append(float(cell))
            except ValueError:
                values.append(np.nan)
        values = np.asarray(values, dtype=np.float64)
        per_condition = {}
        for cond in conditions:
            mask = np.array([l == cond for l in trial_labels])
            vals = values[mask]
            per_condition[cond] = vals[np.isfinite(vals)] / float(scale_factor)
        subjects.append(subject)
        rt_by_subject[subject] = per_condition
    return subjects, trial_labels, rt_by_subject


def select_condition(subjects, rt_by_subject, condition):
    '''
    collapse the per-condition dict to a single array per subject.
    `condition` is either one of the condition labels or 'all' to pool them.
    '''
    out = {}
    for subject in subjects:
        per_condition = rt_by_subject[subject]
        if condition == 'all':
            out[subject] = np.concatenate([per_condition[c] for c in sorted(per_condition)])
        else:
            out[subject] = per_condition[condition]
    return out


def clean_rts(rt, rt_min, rt_max):
    '''
    absolute-window cleaning. Bounds are absolute rather than SD/MAD-based on
    purpose: at ~18 trials the SD is itself dominated by the outlier one would
    be trying to remove, making SD-based trimming circular. The default upper
    bound is deliberately permissive -- the right tail is the effect of
    interest, and the lapse component of the hierarchical model (not trimming)
    is what protects the estimates from contamination.
    '''
    return rt[(rt >= rt_min) & (rt <= rt_max)]


def pooled_reference_quantile(rt_dict, quantile, prefix):
    '''
    pooled quantile over all trials of every subject whose id starts with
    `prefix` (the healthy-control reference sample). Pooling across the whole
    control group means the reference is well estimated even though each
    individual contributes only ~18 trials.
    '''
    pooled = np.concatenate([v for k, v in rt_dict.items() if k.startswith(prefix) and v.size])
    return float(np.percentile(pooled, quantile))


#########################################
#                                       #
#      TAIL-MASS SCORE (AVENUE 2)       #
#                                       #
#########################################


def frac_above(rt, threshold):
    '''
    proportion of a subject's trials exceeding a fixed reference threshold.

    This is the bounded tail-mass score: because it counts trials rather than
    averaging their magnitude, its influence function is bounded -- an
    arbitrarily extreme trial contributes exactly as much as one marginally
    over threshold, so blow-up is structurally impossible and the resulting
    distribution stays continuous (no bimodality).
    '''
    if rt.size == 0:
        return np.nan
    return float(np.mean(rt > threshold))


def residualise_on_speed(tail_scores, median_rts):
    """
    Remove general slowing from a tail score, leaving tail-SPECIFIC slowing.

    A tail-mass score is strongly correlated with how slow a subject is
    overall (r ~ 0.84 with median RT here), so on its own it cannot separate
    "this patient's whole distribution is shifted" from "this patient's right
    tail is disproportionately heavy". Regressing the score on log median RT
    across subjects and keeping the residual isolates the second: a positive
    residual means more tail mass than that subject's overall speed predicts.

    Which of the two is the better lesion marker is an open question, not a
    settled one -- general slowing may well be the more informative signal.
    Reporting the raw score, the speed term and the residual side by side is
    what makes that comparison possible downstream.

    Log median RT is used as the predictor because RT is log-distributed. The
    fit is ordinary least squares, which treats the bounded [0, 1] tail score
    as continuous -- adequate here, but it is a linear approximation to a
    proportion, so residuals for subjects at the 0 floor are least trustworthy.

    Returns (residuals, slope, intercept, r_squared).
    """
    tail = np.asarray(tail_scores, dtype=np.float64)
    x = np.log(np.asarray(median_rts, dtype=np.float64))
    ok = np.isfinite(tail) & np.isfinite(x)
    slope, intercept = np.polyfit(x[ok], tail[ok], 1)
    predicted = slope * x + intercept
    residuals = tail - predicted
    ss_res = np.sum((tail[ok] - predicted[ok]) ** 2)
    ss_tot = np.sum((tail[ok] - tail[ok].mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return residuals, float(slope), float(intercept), float(r_squared)


#########################################
#                                       #
#   SHIFTED 2-PARAMETER FITS (AV. 5)    #
#                                       #
#########################################


def shifted_wald_neg_loglik(params, rt):
    '''
    negative log-likelihood of the shifted Wald (inverse Gaussian first-passage
    time) with drift `gamma`, boundary `alpha` and non-decision shift `theta`.
    Parameterisation follows Anders, Alario & Van Maanen (2016).

    gamma and alpha are passed on a log scale, and theta on a logit scale
    relative to min(rt), so the optimiser is unconstrained and cannot step
    outside the support.
    '''
    log_gamma, log_alpha, logit_theta = params
    gamma, alpha = np.exp(log_gamma), np.exp(log_alpha)
    theta = rt.min() / (1.0 + np.exp(-logit_theta))
    u = rt - theta
    if not np.all(u > 0) or not np.isfinite([gamma, alpha]).all():
        return 1e12
    ll = (np.log(alpha) - 0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(u)
          - (alpha - gamma * u) ** 2 / (2.0 * u))
    total = -np.sum(ll)
    return total if np.isfinite(total) else 1e12


def fit_shifted_wald(rt):
    '''
    ML fit of the shifted Wald. Returns (drift, boundary, shift, converged).

    The drift rate is the parameter of interest: as the first-passage time of
    an evidence-accumulation process it reads directly as accumulation
    efficiency, so a single number carries the severity interpretation that
    motivated tau, with one fewer free parameter to identify.
    '''
    if rt.size < 4:
        return np.nan, np.nan, np.nan, False
    theta0 = 0.9 * rt.min()
    u = rt - theta0
    m, v = u.mean(), u.var(ddof=1)
    if not (m > 0 and v > 0):
        return np.nan, np.nan, np.nan, False
    gamma0 = np.sqrt(m / v)
    alpha0 = m * gamma0
    x0 = [np.log(gamma0), np.log(alpha0), np.log(0.9 / 0.1)]
    res = minimize(shifted_wald_neg_loglik, x0, args=(rt,), method='Nelder-Mead',
                   options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    log_gamma, log_alpha, logit_theta = res.x
    theta = rt.min() / (1.0 + np.exp(-logit_theta))
    ok = bool(res.success) and res.fun < 1e11
    return float(np.exp(log_gamma)), float(np.exp(log_alpha)), float(theta), ok


def shifted_lognormal_profile_neg_loglik(theta, rt):
    '''
    profile negative log-likelihood of the shifted lognormal at shift `theta`.
    For a fixed shift the location and scale have closed-form ML estimates
    (mean and SD of log(rt - theta)), so the whole fit reduces to a stable 1-D
    search over the shift -- no multi-start optimiser, no convergence failures.
    '''
    u = rt - theta
    if not np.all(u > 0):
        return 1e12
    log_u = np.log(u)
    sigma = log_u.std(ddof=0)
    if sigma <= 0:
        return 1e12
    n = rt.size
    total = np.sum(log_u) + n * np.log(sigma) + 0.5 * n * (1.0 + np.log(2.0 * np.pi))
    return total if np.isfinite(total) else 1e12


def fit_shifted_lognormal(rt):
    '''
    ML fit of the shifted lognormal via a 1-D profile search over the shift.
    Returns (shift, mu, sigma, converged), with mu/sigma on the log scale.
    '''
    if rt.size < 4:
        return np.nan, np.nan, np.nan, False
    upper = rt.min() * (1.0 - 1e-6)
    res = minimize_scalar(shifted_lognormal_profile_neg_loglik, args=(rt,),
                          bounds=(0.0, upper), method='bounded',
                          options=dict(xatol=1e-4))
    theta = float(res.x)
    log_u = np.log(rt - theta)
    ok = bool(res.success) and res.fun < 1e11
    return theta, float(log_u.mean()), float(log_u.std(ddof=0)), ok


#########################################
#                                       #
#  HIERARCHICAL EX-GAUSSIAN (AVENUE 1)  #
#                                       #
#########################################


def exgaussian_logpdf(rt, mu, sigma, tau):
    '''
    numpy ex-Gaussian log density, used post-hoc to derive per-trial lapse
    responsibilities from the fitted posterior means (the sampler itself uses
    the PyMC implementation).
    '''
    z = (rt - mu) / sigma - sigma / tau
    return (-np.log(tau) + sigma ** 2 / (2.0 * tau ** 2) - (rt - mu) / tau
            + log_ndtr(z))


def build_hierarchical_exgaussian_lapse(rt, subject_index, n_subjects, lapse_max):
    """
    build the PyMC hierarchical ex-Gaussian + lapse-mixture model

    Each trial is a two-component mixture:
        (1 - w) * ExGaussian(mu_i, sigma_i, tau_i)  +  w * Uniform(0, lapse_max)
    so a contaminated trial is explained by the flat lapse component rather
    than by inflating tau_i -- this is what removes the outlier-driven blow-up
    without deleting the tail.

    PARAMETERISATION. A first version gave every subject their own mu, sigma,
    tau and lapse rate: 4 x n_subjects parameters for ~18 trials each. It did
    not sample -- all chains saturated tree depth, r_hat > 1.01, and mu
    collapsed to a 23 ms spread across subjects whose empirical medians span
    2050 ms, with sigma likewise flat (85.6-94.8). Only tau carried any
    between-subject variance, i.e. mu and sigma were not identified, and
    unidentified parameters are what create the ridges that saturate NUTS.

    This version therefore carries only TWO subject-level parameters:

      total_i  the ex-Gaussian mean, mu_i + tau_i, i.e. that subject's overall
               speed. Well constrained by 18 trials, so sampled CENTRED.
      tail_i   the fraction of that mean contributed by the exponential
               component, tau_i / (mu_i + tau_i), in (0, 1). This is the
               tail parameter of interest -- the disease-severity marker --
               now expressed as a bounded proportion, so it cannot blow up
               the way a free tau can. Weakly constrained, so sampled
               NON-CENTRED.

    mu_i and tau_i are recovered deterministically from the two. Crucially,
    total and tail are close to orthogonal, whereas mu and tau trade off
    directly along a ridge (their sum is pinned by the data while the split
    between them is not) -- that ridge was the geometry problem.

    sigma is reduced to a single group-level coefficient of variation, sigma_i
    = sigma_cv * total_i, rather than a free per-subject parameter. The data
    supported essentially one value, but a single absolute sigma across
    subjects ranging 200-2250 ms would be untenable, so it scales with speed.

    The lapse rate is likewise a single group-level w: an individual lapse
    rate is not identified from ~18 trials. Per-subject lapse RESPONSIBILITY
    is still reported, derived post-hoc from the fitted densities rather than
    sampled as a free parameter.

    NOTE: lapse_max must span the observed data. The Uniform component is what
    gives extreme trials a non-zero density; setting it below max(rt) makes
    both mixture components zero there and the likelihood -inf.

    `rt` is the concatenated trial vector, `subject_index` the matching
    0-based subject index per trial.
    """
    import pymc as pm
    import pytensor.tensor as pt

    lapse_max = max(float(lapse_max), float(rt.max()) * 1.01)

    with pm.Model() as model:
        # ---- group level -------------------------------------------------
        log_total_mu  = pm.Normal('log_total_mu', mu=np.log(np.median(rt)), sigma=1.0)
        log_total_sd  = pm.HalfNormal('log_total_sd', sigma=1.0)
        logit_tail_mu = pm.Normal('logit_tail_mu', mu=-1.0, sigma=1.0)
        logit_tail_sd = pm.HalfNormal('logit_tail_sd', sigma=1.0)
        sigma_cv      = pm.HalfNormal('sigma_cv', sigma=0.5)
        w             = pm.Beta('w', alpha=2.0, beta=200.0)

        # ---- subject level -----------------------------------------------
        # total is well informed -> centred; tail is weakly informed -> non-centred
        total_i = pm.LogNormal('total_i', mu=log_total_mu, sigma=log_total_sd,
                               shape=n_subjects)
        z_tail  = pm.Normal('z_tail', mu=0.0, sigma=1.0, shape=n_subjects)
        tail_i  = pm.Deterministic('tail_i',
                                   pm.math.sigmoid(logit_tail_mu + logit_tail_sd * z_tail))

        tau_i   = pm.Deterministic('tau_i',   total_i * tail_i)
        mu_i    = pm.Deterministic('mu_i',    total_i * (1.0 - tail_i))
        sigma_i = pm.Deterministic('sigma_i', sigma_cv * total_i)

        # ---- trial-level mixture likelihood ------------------------------
        weights = pt.stack([pt.ones_like(rt) * (1.0 - w), pt.ones_like(rt) * w], axis=1)
        pm.Mixture('obs',
                   w=weights,
                   comp_dists=[pm.ExGaussian.dist(mu=mu_i[subject_index],
                                                  sigma=sigma_i[subject_index],
                                                  nu=tau_i[subject_index]),
                               pm.Uniform.dist(lower=0.0, upper=lapse_max)],
                   observed=rt)
    return model
