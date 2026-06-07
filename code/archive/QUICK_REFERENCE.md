# Quick Reference: Redundant Multilayer SBM Implementation

## What Was Done

Implemented a weighted nested stochastic block model (SBM) that:
- ✓ Detects hierarchical community structure
- ✓ Handles two separate layers with different edge semantics
- ✓ Preserves edge weights for inference via Bayesian likelihoods
- ✓ No shape mismatch errors during initialization

## Key Insight: Redundant Edges

Instead of trying to make a single edge belong to two layers simultaneously, each edge appears **twice**:

```
Physical edge (node_i, node_j):
├─ Instance in layer 0: weight for layer 0 active, layer 1 weight = 0
└─ Instance in layer 1: weight for layer 1 active, layer 0 weight = 0
```

Benefits:
- Simplifies LayeredBlockState initialization
- Both layers always populated (prevents numerical issues)
- Weights remain associated with correct layer during inference
- No modification to layer property semantics

## Critical Fix: Omit rec_params

When initializing `LayeredBlockState` as a base for `NestedBlockState`, **do NOT** specify custom priors:

```python
# WRONG - causes shape error
state_args=dict(
    base_type=gt.LayeredBlockState,
    state_args=dict(
        ec=g.ep.layer,
        recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
        rec_types=["real-normal", "discrete-poisson"],
        rec_params=[dict(m0=0., k0=1, v0=1., nu0=3), dict(alpha=1, beta=1.)],  # ← DELETE
        layers=True,
        deg_corr=True
    )
)

# RIGHT - uses graph-tool defaults
state_args=dict(
    base_type=gt.LayeredBlockState,
    state_args=dict(
        ec=g.ep.layer,
        recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
        rec_types=["real-normal", "discrete-poisson"],
        # rec_params omitted
        layers=True,
        deg_corr=True
    )
)
```

## Testing

Run the comprehensive test to verify everything works:

```bash
cd /mnt/h/GitHub/rtm/code
python sbm/test_full_implementation.py
```

Expected output:
- Graph creation: ✓
- Redundant edge structure: ✓
- Weight complementarity: ✓
- NestedBlockState initialization: ✓
- MCMC sampling: ✓
- Block structure extraction: ✓

## Files Changed

| File | Function | Changes |
|------|----------|---------|
| `get_graphobject.py` | `create_multilayer_graph()` | Creates redundant edges (2× node pairs) |
| `sbm_community_detection_weight.py` | `fit_nested_sbm_layered()` | Removed rec_params, fixed array conversion |

## The Math

**Graph Structure:**
- Nodes: 166 (brain regions)
- Edges: 2 × (unique node pairs with edges in either layer)
- Example: 12,016 unique pairs → 24,032 edges (12,016 in each layer)

**Inference Model:**
```
Layer 0 (BEHAVIOUR):
  P(weight | block_assignment) ∝ exp(-(weight - μ)² / σ²)  [Gaussian]

Layer 1 (Co-occurrence):
  P(count | block_assignment) ∝ (λ^count * e^-λ) / count!   [Poisson]

NestedBlockState:
  Combines both layers in hierarchy via Bayesian inference
  Estimates block structure at multiple resolution levels
```

## No Breaking Changes

- ✓ Layer property semantics unchanged (0=behaviour, 1=cooccurrence)
- ✓ Backward compatible with existing node/edge properties
- ✓ API of fit_nested_sbm_layered() unchanged
- ✓ Returns same result dictionary structure

## Performance Notes

- Edge doubling increases memory ~2× (manageable for 166 nodes)
- MCMC sampling complexity unchanged (per-node computations)
- Hierarchy construction slightly more expensive (full redundancy to process)
- Gains: Stable initialization, no shape errors, reliable inference
