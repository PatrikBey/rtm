# ROI Selection: Joint Lesion-Behaviour Information

Goal: identify ROIs whose block structure in the **observed** multilayer SBM fit
reflects genuine lesion-behaviour coupling — not something explainable by chance
(behaviour-permutation null) or by lesion topology alone (lesion-only fit).

Three output groups feed this:
- **lesion-only fit** — SBM on the cooccurrence layer alone
- **multilayer + permuted behaviour** — `run_null.py` output (100 perms, `SBMNULL/`)
- **multilayer + observed behaviour** — `run.py` output (the real fit)

Approaches 1-3 below are complementary filters; the entropy check is a global
sanity gate, not a per-ROI filter. Approach 1 is implemented (see below);
Approaches 2-3 are outline only.


## Approach 1: Null-referenced significance

**Idea:** ask whether an observed lvl0 grouping's behavioural coherence is more
extreme than what the same SBM fit produces when behaviour is randomly
shuffled. The null already holds lesion cooccurrence structure fixed and only
breaks the true behaviour-lesion pairing, so a large deviation from that null
is specific evidence of the real pairing, not an artefact of lesion topology
or the SBM's general behaviour.

Implemented for all three available tasks (`Foreperiod_Long_tau`, `GoNoGo_tau`,
`SATO_Accuracy_tau`) — `SATO_Accuracy_tau` was added in a later run than the
other two but through the identical pipeline (observed fit, lesion-only base
fit, behaviour-permutation null), so it is an equally valid third task, not a
provisional/pilot result.
Scripts live in `code/analysis/`; outputs in `{data_path}/ROISELECTION/`. The
**current, canonical form tests at the block level** (below) — an earlier
per-ROI version was tried first and is kept here as a documented dead end,
since it explains why the block-level design looks the way it does.

### Current implementation: block-level, max-statistic (`block_pvalues.py`)

Tests each SBM block once, using its raw composite score rather than a
per-ROI or z-scored value:

- **Raw score, not z-score.** Uses the "score" column from
  `SBM_block_scores_lvl{level}_{task}.csv` (consistency-weighted mean
  `behaviour_degree` per block, i.e. `np.average(behaviour_degree,
  weights=clip(node_consistency, 0))`) rather than the z-scored version. The
  z-score re-normalizes across blocks *within one fit*, which isn't needed for
  a permutation-based comparison and only adds fit-dependent noise (block
  count varies from fit to fit).
- **Max-statistic null, not matched block ids.** Block ids aren't comparable
  across fits — each SBM fit (observed or permuted) discovers its own
  independent partition, so "block 3" in permutation 47 has no relation to
  "block 3" in the observed fit. Forcing a permutation's null value onto the
  *observed* block's specific ROI membership was tried and rejected: that
  membership was chosen by fitting on the real behaviour scores, so evaluating
  random behaviour over that same fixed set is double-dipping / selection
  bias — it never lets the null's own discovery process compete on equal
  footing.

  Instead, for each permutation, take the single highest raw block score
  among *that permutation's own* freely-discovered blocks:

  ```
  null_max = [max(block_scores_of(perm)) for perm in null_permutations]   # 100 values

  for block in observed_blocks:
      p[block] = (count(null_max >= block.score) + 1) / (n_perm + 1)
  ```

  `null_max` is the sampling distribution of "the most behaviourally-coherent
  grouping chance alone can produce" for that task — already inflated by the
  fact that *some* block is always going to look best out of ~12-13 by chance,
  which is exactly the effect that needs correcting for. Because every
  observed block is compared against this same max-derived null, the test is
  family-wise-error-controlled by construction; no separate BH-FDR step is
  applied on top.

**Result: 2/12 blocks significant (p < 0.05) for Foreperiod_Long_tau, 2/13 for
GoNoGo_tau, 0/13 for SATO_Accuracy_tau** — SATO_Accuracy_tau's own
free-partition null test found no block whose behavioural coherence exceeded
chance (minimum p = 0.24, block 11), so it is a genuine null result under this
test rather than a gap in coverage. Per-task tables
(`block_pvalues_lvl0_{task}.tsv`) and significant-block NIfTIs
(`{atlas}_lvl0_significant_blocks_{task}.nii.gz`, observed score kept only for
significant blocks' member ROIs) are in `ROISELECTION/`; all three tasks'
significant-block NIfTIs are also copied into `doc/` (SATO_Accuracy_tau's is
all-zero, reflecting the null result).

### Superseded: per-ROI z-score + BH-FDR (`roi_pvalues.py`, `roi_pvalues_lvl1.py`)

The original design tested every ROI individually:

```
observed_z   = load_column(roi_block_assignments_observed.csv, "zscore_0")  # per ROI
null_z       = load_table(permutation_zscores.tsv)                          # roi x 100

for roi in rois:
    p[roi] = (count(null_z[roi] >= observed_z[roi]) + 1) / (n_perm + 1)

p_fdr = benjamini_hochberg(p)
hits_1 = rois where p_fdr < alpha
```

Still implemented (lvl0 and lvl1) and kept for reference, but **not the
current recommendation** — it produced **0/400 ROIs significant at FDR < 0.05,
at both levels, for both tasks**, even though GoNoGo_tau lvl0 had 24/400 ROIs
with raw p < 0.05 before correction. Two compounding issues drove this:

- **Pseudo-replication.** A block's z-score is broadcast identically to every
  ROI it contains, so BH-FDR across 400 "ROIs" is really correcting for only
  ~11-20 distinct values (one per block) repeated many times over — far more
  punishing than warranted. This is what motivated testing at the block level
  instead.
- **Permutation floor.** With only 100 permutations, the smallest possible raw
  p-value is `1/101 ≈ 0.0099`, which limits how far BH-FDR can push any
  p-value down regardless of effect size.


## Approach 2: Residualize against the lesion-only layer

**Idea:** per ROI, behavioural connectivity (`behaviour_weight` edges) may
simply track how often the ROI is lesioned at all (`cooccurrence_weight`
edges / lesion-only block score). Regress the behavioural signal on the
lesion-only signal across ROIs and keep the residual — ROIs with behavioural
connectivity in excess of what lesion frequency predicts carry information
beyond pure lesion occurrence.

Uses per-node `beh_degree` / `occ_degree` (already computed in
`_save_permutation_outputs`, same quantities available from the observed run)
or the lesion-only fit's block score as the predictor.

```
x = occ_degree_or_lesion_only_score   # per ROI, predictor
y = beh_degree_observed               # per ROI, outcome

slope, intercept = linear_fit(x, y)
residual[roi] = y[roi] - (slope * x[roi] + intercept)

hits_2 = rois where residual > threshold   # e.g. top decile, or z(residual) > 2
```


## Approach 3: Partition comparison — lesion-only vs. observed multilayer

**Idea:** if a ROI's block membership is unchanged between the lesion-only fit
and the observed multilayer fit, its grouping is being driven by lesion
topology alone. ROIs that reorganize when the behaviour layer is added are
gaining genuinely new (joint) structure. Weight by `node_consistency` so
unstable/ambiguous nodes in either fit don't inflate the signal.

```
b_lesion  = modal_assignments_lesion_only[level]      # per ROI block id
b_multi   = modal_assignments_observed[level]         # per ROI block id
c_lesion  = node_consistency_lesion_only[level]
c_multi   = node_consistency_observed[level]

for roi in rois:
    same_block = neighbours(b_lesion, roi) == neighbours(b_multi, roi)  # co-membership overlap
    reorg[roi] = 1 - jaccard(same_block)
    weight[roi] = c_lesion[roi] * c_multi[roi]

reorg_weighted = reorg * weight
hits_3 = rois where reorg_weighted > threshold
```

(`neighbours(b, roi)` = set of other ROIs sharing `roi`'s block; comparing
co-membership sets rather than raw block ids avoids label-switching issues
between the two independently-fit partitions.)


## Global check: entropy / description-length validation

**Idea:** before trusting any per-ROI list, confirm the observed fit actually
compresses the data better than chance. Compare the observed model's
description length against the null distribution of entropies (`results['entropy']`,
already saved per permutation in `SBMNULL/.../perm_XXXXX/`). If the observed
entropy isn't an outlier relative to the null distribution, the per-ROI hits
above are more likely noise than signal, regardless of how many pass Approach 1.

```
entropy_observed = load(observed_run, "entropy")
entropy_null     = [load(perm, "entropy") for perm in null_permutations]   # 100 values

p_global = (count(entropy_null <= entropy_observed) + 1) / (n_perm + 1)
# lower entropy = better fit -> observed should sit in the tail

if p_global >= alpha:
    flag("global fit not distinguishable from null — treat ROI hits cautiously")
```


## Combining evidence

Suggested read, from cheapest/most direct to most conservative:

1. Run the global entropy check first — a sanity gate, not a filter.
2. Rank ROIs by Approach 1 (null-referenced significance) — the primary,
   already-supported test.
3. Intersect with Approach 2 and/or Approach 3 to require the effect isn't
   just lesion occurrence riding along — `hits_1 ∩ (hits_2 ∪ hits_3)` is a
   reasonable conservative joint-information set.
4. Optionally weight by cross-level robustness (repeat Approach 1 at lvl1-lvl3
   and require agreement across ≥2 levels) before finalizing the list.
