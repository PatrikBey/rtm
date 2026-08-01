# ROI Selection: Joint Lesion-Behaviour Information

Goal: identify ROIs whose block structure in the **observed** multilayer SBM fit
reflects genuine lesion-behaviour coupling — not something explainable by chance
(behaviour-permutation null) or by lesion topology alone (lesion-only fit).

Three output groups feed this:
- **lesion-only fit** — SBM on the cooccurrence layer alone
- **multilayer + permuted behaviour** — `run_null.py` output (100 perms, `SBMNULL/`)
- **multilayer + observed behaviour** — `run.py` output (the real fit)

Approaches 1-3 below are complementary filters; the entropy check is a global
sanity gate, not a per-ROI filter. Not implemented here — outline only.


## Approach 1: Null-referenced significance per ROI

**Idea:** for each ROI, ask whether its observed lvl0 block z-score is more
extreme than the 100 z-scores obtained when behaviour was randomly shuffled.
The null already holds lesion cooccurrence structure fixed and only breaks the
true behaviour-lesion pairing, so a large deviation from that null is specific
evidence of the real pairing, not an artefact of lesion topology or the SBM's
general behaviour.

Uses `permutation_zscores.tsv` (roi x permutation, already generated) plus the
observed run's `roi_block_assignments_{score}.csv` (`zscore_0` column).

```
observed_z   = load_column(roi_block_assignments_observed.csv, "zscore_0")  # per ROI
null_z       = load_table(permutation_zscores.tsv)                          # roi x 100

for roi in rois:
    p[roi] = (count(null_z[roi] >= observed_z[roi]) + 1) / (n_perm + 1)

p_fdr = benjamini_hochberg(p)
hits_1 = rois where p_fdr < alpha
```


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
