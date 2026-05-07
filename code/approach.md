# How `pylater` Implements the LATER Model

## The LATER Model (Noorani & Carpenter, 2016)

LATER (Linear Approach to Threshold with Ergodic Rate) is a model of reaction time (RT) based on a simple idea: before each trial, a decision signal rises linearly from a starting level toward a fixed threshold. The rate of rise `r` is drawn fresh from a **Gaussian distribution** on each trial, giving RT = threshold / r. Crucially, rather than modelling RT directly, the model works in **promptness** space (λ = 1/RT), where the distribution is simply normal.

The full model includes two competing processes that race to threshold:

| Process | Signal | Meaning |
|---------|--------|---------|
| **Late (main)** | N(μ, σ) in promptness | Deliberate, cortically-driven decision |
| **Early** | N(0, σ_e) in promptness | Fast, pre-attentive trigger (zero-mean) |

Whichever process reaches threshold first determines RT. This produces the characteristic **reciprocal-normal** distribution visible as a straight line on a *reciprobit* plot (x-axis = 1/RT, y-axis = cumulative probability on a probit scale).

## How `pylater` Implements This

### Distribution (`dist.py`)

The log-probability of an observed RT (converted to promptness `r = 1/RT`) is derived from the race:

```
log p(r) = log[ f_late(r)·Φ_early(r) + f_early(r)·Φ_late(r) ]
```

where `f` is the Gaussian PDF and `Φ` is the Gaussian CDF.  This is a `logsumexp` of two terms — one for each possible "winner" of the race.

Simulation mirrors this: draw one sample from each Gaussian, take the maximum (fastest / highest promptness), return 1/max as RT.

### Parameters

Rather than `μ` and `σ` directly, the model builder (`model.py`) uses:

| Fitted parameter | Meaning |
|-----------------|---------|
| `sigma` | Std dev of main Gaussian in promptness |
| `k = mu/sigma` | Signal-to-noise ratio (proportional shift of the mean) |
| `sigma_e_mod = sigma_e/sigma` | Early noise relative to main noise |

`mu` and `sigma_e` are then deterministic: `mu = k · sigma`, `sigma_e = sigma_e_mod · sigma`.  All three free parameters have LogNormal priors.

### Bayesian inference

`build_default_model()` wraps everything in a **PyMC model**, so fitting is full Bayesian MCMC via `pm.sample()`. Outputs are posterior distributions over the three parameters, plus prior/posterior predictive RTs in seconds.

### Multi-dataset sharing

When fitting multiple conditions simultaneously, two sharing modes are available:

- **Shift**: conditions share `sigma` but have separate `k` — the reciprobit lines are parallel (same slope, different intercept).
- **Swivel**: conditions share `k` but have separate `sigma` — the lines pivot around a common point (same intercept ratio, different slope).

### Visualisation

`ReciprobitPlot` renders data and model fits on a reciprobit axes (reciprocal time on x, probit probability on y), which linearises the LATER distribution. A straight fitted line confirms the model's Gaussian-promptness assumption.
