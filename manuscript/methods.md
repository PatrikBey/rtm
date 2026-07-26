# Methods

*Draft methods section — RTM (Reaction Time Modelling) project. Covers the two
stochastic block modelling (SBM) frameworks implemented in
[`code/sbm/run.py`](../code/sbm/run.py) / [`code/sbm/functions.py`](../code/sbm/functions.py)
and [`code/sbm/run_recon_pnb.py`](../code/sbm/run_recon_pnb.py), and the figures
produced by [`code/visualizations/create_figures.py`](../code/visualizations/create_figures.py)
and [`code/visualizations/create_figures_pnb.py`](../code/visualizations/create_figures_pnb.py).
Bracketed citations are placeholders for the corresponding graph-tool /
Peixoto references and should be verified against the exact software version
used before submission. Sample-size, scanner/lesion-mapping, and behavioural
task descriptions are outside the scope of these two scripts and are marked
`[TODO]` below.*

---

## 1. Overview

Two complementary, non-overlapping modelling strategies were used to relate
patient-level lesion topography to behavioural performance across three
reaction-time tasks (`Foreperiod_Long_tau`, `GoNoGo_tau`, `SATO_Accuracy_tau`;
Long Foreperiod, Go/No-Go, and Simple/Sustained-Attention task-derived τ
parameters `[TODO: task battery citation]`). Both strategies infer a
*nested* stochastic block model (SBM) — a hierarchical, degree-corrected
generative model of network community structure `[Peixoto, Phys. Rev. X 4,
011047, 2014]` — implemented in `graph-tool` `[Peixoto, arXiv:1408.0553]`, but
they differ in what is treated as the observed network:

1. **Framework I — layered co-occurrence/behaviour SBM** (`run.py`,
   `functions.py`): a two-layer graph is built explicitly from
   pairwise lesion co-occurrence and behaviour-weighted co-occurrence across
   patients, thresholded into a fixed edge set, and a joint nested SBM with
   edge covariates is fit to that fixed graph via `gt.LayeredBlockState`
   `[Peixoto, Phys. Rev. E 92, 042807, 2015]`.
2. **Framework II — continuous network reconstruction** (`run_recon_pnb.py`):
   no graph is constructed by hand; instead, the region-by-region network is
   itself treated as *latent* and reconstructed directly from continuous
   per-subject regional lesion-load values via `gt.PseudoNormalBlockState`,
   a pseudo-likelihood reconstruction model that jointly infers edge
   couplings and block structure from node-level observations
   `[Peixoto, Phys. Rev. X 8, 041011, 2018; Peixoto, Phys. Rev. Lett. 123,
   128301, 2019]`.

Both frameworks are fit per behavioural task, use the Schaefer 400-region
cortical parcellation (`Schaefer2018-400`) `[Schaefer et al., 2018]` as the
common node space, use finite-temperature Markov chain Monte Carlo (MCMC)
with an automated mean-shift change-point criterion to decide when the chain
has converged, and summarise the post-convergence posterior into a modal
block partition together with a chance-corrected per-node assignment
consistency. They differ in how behaviour enters the model (as an explicit
graph layer in Framework I; as a multiplicative weight on the input data in
Framework II) and in how blocks are related back to behaviour for reporting
(a consistency-weighted mean nodal degree in Framework I; a direct
node–behaviour Spearman correlation, "relevance", in Framework II).

## 2. Data

`[TODO — not defined in run.py / run_recon_pnb.py; complete from
acquisition/preprocessing methods]`

- **Participants**: `participants.tsv`; three behavioural τ scores per
  subject (`Foreperiod_Long_tau`, `GoNoGo_tau`, `SATO_Accuracy_tau`).
  Subjects with a missing score for the task under analysis were excluded
  from that task's model (`subjects_missing_score`, logged per run).
- **Parcellation**: Schaefer 400-region cortical atlas (`Schaefer2018-400`),
  co-registered to each subject; ROI metadata (`{atlas}_areas.txt`) supplies
  node labels and coarse anatomical group/hemisphere assignment used for
  figure legends.
- **Framework I input**: per-subject binary disconnectome matrices
  (`DISCONNECTOMES/{subject}_{atlas}.tsv`), thresholded at the subject's own
  50th percentile of non-zero streamline values (`load_graphs`,
  [`utils.py:47`](../code/sbm/utils.py)) to obtain a binary region × region
  disconnection matrix per subject; subjects with an all-zero disconnectome
  were excluded (`empty_subjects`).
- **Framework II input**: a subject × region continuous lesion-load matrix
  (`{atlas}_lesion_loads.tsv`, values in [0, 100]), used directly with no
  binarisation. Subjects with zero variance across all regions were masked
  out (flagged as a likely loading artefact; `load_data`,
  [`run_recon_pnb.py:433`](../code/sbm/run_recon_pnb.py)).

## 3. Framework I — Layered nested SBM of lesion co-occurrence and behaviour

Implemented in [`run.py`](../code/sbm/run.py) (pipeline driver) and
[`functions.py`](../code/sbm/functions.py) (`create_multilayer_graph`,
`fit_nested_sbm_layered[_multiflip]`).

### 3.1 Multilayer graph construction

For each task, two undirected region × region matrices were constructed
across the included patient sample (`create_multilayer_graph`,
[`functions.py:38`](../code/sbm/functions.py)):

- a **behaviour-weighted co-occurrence layer**, obtained by summing each
  patient's binary disconnectome matrix scaled by that patient's raw
  behavioural score, $W^{beh} = \sum_p A_p \cdot y_p$; and
- a **co-occurrence layer**, the unweighted patient-wise sum of binary
  disconnectome matrices, $W^{occ} = \sum_p A_p$.

Each layer was thresholded independently at its own edge-density percentile
(default: 75th percentile of non-zero edge weights; `--edge_threshold`),
setting sub-threshold entries to zero. By default (`--combined-layers`,
enabled unless `--no-combined-layers` is passed) the two layers' surviving
edge masks were intersected, so both layers share an identical edge set and
differ only in edge weight — this yields one shared graph topology on which
the two covariates (behaviour-weighted co-occurrence, raw co-occurrence) are
modelled jointly, rather than allowing each layer to independently
determine which node pairs are connected. If the cooccurrence layer is
modelled with a Poisson edge distribution (`--cooccurrence_dist poisson`),
its weights are additionally min–max rescaled to [0, 1] before thresholding,
since the discrete-Poisson description length is not scale-invariant.

The resulting `graph-tool` graph carries, per edge, a `layer` indicator
(0 = behaviour-weighted co-occurrence, 1 = co-occurrence) and the two raw
edge weights (`behaviour_weight`, `cooccurrence_weight`) as internal edge
properties, and is saved (`SBM_graph_{task}.gt`) together with each layer's
dense adjacency matrix (`SBM_layer_{task}_{behaviour,cooccurrence}.txt`) for
downstream figure generation.

### 3.2 Model specification

The multilayer graph was modelled as a degree-corrected nested SBM with
edge covariates via `gt.LayeredBlockState` (`state_args=dict(ec=layer,
recs=[behaviour_weight, cooccurrence_weight], rec_types=[...],
layers=True, deg_corr=True)`), initialised by `graph-tool`'s greedy
agglomerative heuristic, `minimize_nested_blockmodel_dl`
`[Peixoto, Phys. Rev. X 4, 011047, 2014]`. Each layer's edge weight was
modelled with its own conditional distribution (default: real-normal for
both layers, i.e. scale-invariant; discrete-Poisson is available for the
co-occurrence layer via `--cooccurrence_dist poisson`), so a single joint
block partition is inferred that must simultaneously explain both the
weighted-co-occurrence and raw co-occurrence structure — this is what
"joint" means throughout: one partition, two edge covariates.

### 3.3 Posterior inference

Given the initial partition, the model was refined by finite-temperature
MCMC. Two proposal kernels were implemented and are selectable via
`--multiflip`:

- **single-flip** (`state.mcmc_sweep`, default): local, single-vertex
  Metropolis–Hastings moves;
- **merge-split / multiflip** (`state.multiflip_mcmc_sweep`,
  `--multiflip`): proposes block merge and split moves in addition to
  single-vertex moves, which can escape local optima that single-vertex
  moves cannot `[Peixoto, Phys. Rev. E 102, 012305, 2020]`.

Convergence was assessed online via a **mean-shift change-point rule**
applied to the per-sweep model entropy (description length, DL) trajectory,
common to both proposal kernels (`fit_nested_sbm_layered[_multiflip]`,
[`functions.py:150`](../code/sbm/functions.py)):

1. The first `window_size` sweeps (default 250) establish a reference mean
   $\mu_0$ and standard deviation $\sigma_0$ of the entropy trajectory.
2. From sweep `window_size` onward, a trailing sliding-window mean of the
   last `window_size` entropy values is compared against the threshold
   $\theta = \mu_0 - \gamma\,\sigma_0$, where $\gamma$ is `--shift_factor`
   (default 0.75).
3. The **change point** is the first sweep at which the sliding-window mean
   drops below $\theta$, i.e. a sustained downward shift in model entropy
   relative to the initial-window scale of variability. If no change point
   is detected within `--max_iter` sweeps (default 10,000), a warning is
   logged and the final chain state is used instead.

Following the detected change point (or exhaustion of `max_iter`), a fixed
**accumulation phase** of `window_size` additional sweeps was run from the
current state, during which block assignments at every meaningful hierarchy
level (levels with more than one non-empty block relative to the
lowest-entropy level; `meaningful_levels`), the level-0 block-to-block edge
count matrix (`mrs`), and their first two raw moments were recorded at every
sweep.

### 3.4 Posterior summarisation

From the accumulation-phase samples:

- **Modal partitions.** For each meaningful level $k$, the sequence of
  sampled partitions (each level-0 partition re-projected up to level $k$
  via `state.project_partition`) was passed to `gt.PartitionModeState`
  (`converge=True`), which resolves label-switching across samples and
  returns the single partition maximising posterior probability
  (`get_max`) — the *modal* partition reported for that level, not the
  chain's final-iteration state nor a simple average.
- **Node assignment consistency.** For each node, the posterior marginal
  probability of its own modal block (`get_marginal`) was converted to a
  chance-corrected consistency analogous to Cohen's κ:
  $\kappa = (p_{modal} - 1/B)/(1 - 1/B)$, where $B$ is the number of blocks
  at that level. $\kappa = 1$ indicates the node was assigned to its modal
  block on every sample; $\kappa = 0$ indicates chance-level consistency;
  $\kappa < 0$ indicates below-chance consistency. This corrects for the
  fact that, with many blocks, even a modest raw marginal probability can
  represent strong evidence relative to the chance baseline of $1/B$.
- **Joint block connectivity.** For each meaningful level, the level-0
  block-to-block edge-count matrices sampled during accumulation were
  re-aggregated into that level's block space (via level-0 → level-$k$
  block mapping and majority-vote remapping to the modal level-$k$
  labelling) and averaged, yielding a single $B \times B$ matrix
  (`block_connectivity[k]`) that reflects the posterior-expected joint
  edge density between communities under the fitted model — not a
  post-hoc projection of the original raw layer weights onto blocks.
- **Posterior edge moments.** The mean and variance of each node's
  own-block internal edge mass across accumulation samples
  (`edge_mean`, `edge_var`) were also retained per node.

### 3.5 Spatial block mapping and selection

To identify which communities were informative about the joint
lesion/behaviour structure, each block at each meaningful level was scored
by the consistency-weighted mean behavioural-layer weighted degree of its
member nodes (negative consistency values clipped to zero so unreliable
nodes cannot invert the weighting). Block scores were z-scored within each
level, and only blocks with above-average score (z > 0) were treated as
behaviourally relevant and written out individually. For every level, a
per-ROI CSV table (`roi_block_assignments_{task}.csv`; ROI name, per-layer
weighted degree, own-block posterior edge mean/variance, block ID and
consistency per meaningful level) and, per level, a NIfTI volume in which
every voxel is replaced by its block's z-score (`{atlas}_lvl{k}_
blockzscores_{task}.nii.gz`) were written, together with one NIfTI + text
region-mapping file per selected block and the level's raw block
connectivity matrix (CSV).

## 4. Framework II — Continuous PseudoNormalBlockState reconstruction

Implemented in [`run_recon_pnb.py`](../code/sbm/run_recon_pnb.py). Unlike
Framework I, no graph is constructed from patient-level disconnectomes;
instead, the region × region network is treated as an unobserved latent
variable and reconstructed directly from a subject × region matrix of
continuous lesion-load values, with no binarisation step at any stage.

### 4.1 Data preparation and behavioural weighting

Per task, subjects with a missing behavioural score were dropped, and
subjects with zero variance across all regions were masked out as likely
loading artefacts (`load_data`, [`run_recon_pnb.py:433`](../code/sbm/run_recon_pnb.py)).
The remaining subject × region lesion-load matrix was globally min–max
rescaled to [0, 1] (`rescale_01`). Two input variants were then fit per
task:

1. **`no_beh`** — the rescaled lesion-load matrix, unweighted.
2. **`beh_weighted`** — the rescaled lesion-load matrix multiplied
   row-wise (per subject) by that subject's own min–max-rescaled
   behavioural score, amplifying each subject's lesion pattern in
   proportion to their behavioural score before reconstruction.

Appending behaviour as an additional, separately correlated pseudo-node was
also tested during development and found to never change the recovered
block structure relative to `no_beh` — this SBM groups nodes by pairwise
correlation *profile*, and has no mechanism to condition that grouping on a
third variable appended as a node — so multiplicative weighting was used
instead, as the only mechanism found empirically to alter recoverable
structure relative to the behaviour-blind fit.

Both variants were transposed to region × subject form ($S$) as required by
`gt.PseudoNormalBlockState`.

### 4.2 Model specification

Each variant was modelled with `gt.PseudoNormalBlockState(S, nested=True)`,
a nested, degree-corrected pseudo-likelihood network reconstruction model
that jointly infers, from a matrix of per-node continuous observations
across independent samples (here: per-region lesion load across subjects),
(i) a latent region × region coupling graph with continuous edge strengths
$x_{ij}$, and (ii) a nested block partition of the regions, under a
Gaussian (real-normal) pseudo-likelihood linking coupling strength and
partition to the observed data
`[Peixoto, Phys. Rev. X 8, 041011, 2018; Peixoto, Phys. Rev. Lett. 123,
128301, 2019]`.

### 4.3 Posterior inference

Fitting used the same mean-shift change-point loop structure as Framework I
(`fit_pseudo_normal`, [`run_recon_pnb.py:99`](../code/sbm/run_recon_pnb.py)),
with `window_size` = 250 sweeps (default), `shift_factor` = 0.5 (default),
and `max_iter` = 10,000 (default). Critically, **only plain single-flip
`state.mcmc_sweep()` was used, never `multiflip_mcmc_sweep()`.** This is the
reverse of the recommendation for Framework I / general nested-SBM fitting:
in controlled comparisons on synthetic data with known ground-truth block
structure, multiflip search on `PseudoNormalBlockState` failed to recover
known structure at all (best adjusted mutual information ≈ 0.03–0.08 across
2,000 sweeps), whereas single-flip search reliably and reproducibly
recovered it (AMI = 1.000) across seeds and across the realistic density
range tested (see project development notes,
`project_run_recon_ising_fit_investigation`). No accumulation/posterior-
averaging phase analogous to Framework I's is performed here; the converged
(or `max_iter`-exhausted) state is used directly to extract the latent
graph and block partition. The fitted coupling strength for each latent
edge is retained as an edge property (`x`) when the graph is extracted
(`extract_graph`), and block assignments at level 0 (base partition) and
level 1 are obtained by projecting the nested partition down via
`NestedBlockState.project_partition` (`get_blocks`).

### 4.4 Region–behaviour relevance and block selection

Because Framework II's block partition is not fit jointly with an explicit
behaviour covariate (behaviour only enters, indirectly, via the
`beh_weighted` input multiplication), each region's association with
behaviour was quantified independently as the Spearman rank correlation
between that region's raw (unrescaled) per-subject lesion-load values and
the raw behavioural score (`region_behaviour_relevance`; regions with zero
variance in-sample assigned relevance 0). This region–behaviour relevance
is shared across both input variants for a given task (it does not depend
on `no_beh` vs. `beh_weighted`) and is used both for block selection and
for figure colouring (§6).

For each fitted variant and hierarchy level (0 and 1), blocks were scored
by their members' mean relevance, z-scored across blocks, and blocks with
above-average relevance (z > 0) were selected as behaviourally relevant
(`select_relevant_blocks`) — the same z > 0 selection logic used in
Framework I, applied here to relevance rather than to consistency-weighted
degree. For each selected block, a NIfTI volume marking its member regions
was written (`save_block_niftis`); a single combined NIfTI covering all
regions, each voxel carrying its own block's mean relevance value
regardless of selection, was also written (`save_block_value_nifti`); and a
per-ROI TSV summary (block ID, relevance, block score/z-score, selection
flag) was saved (`save_roi_assignment_table`).

### 4.5 Visualisation

The fitted (possibly nested) block state for each variant was rendered with
`graph-tool`'s hierarchy drawing (`draw_state`,
[`run_recon_pnb.py:337`](../code/sbm/run_recon_pnb.py)): node colour encodes
signed region–behaviour relevance via a diverging colormap; edge colour is
the average of its endpoints' colours, with edges below the 90th percentile
of endpoint-averaged |relevance| rendered fully transparent and the
remainder scaled in alpha by relevance magnitude (γ = 4 power transform),
so that only the most behaviourally relevant coupling structure is visually
emphasised.

## 5. Software

`graph-tool` `[Peixoto, arXiv:1408.0553]` (version `[TODO]`), Python
`[TODO]`, `numpy`, `nibabel`, `scipy.stats.spearmanr`, `tqdm`. All MCMC runs
used a fixed random seed (`--seed`, default 42) passed to
`gt.seed_rng` for reproducibility. Runs were performed independently per
behavioural task (and, for Framework II, per input variant), producing one
output directory per task under `RESULTS/`.

---

## 6. Figures

Two figure-generation scripts produce the manuscript figures, one per
framework: [`create_figures.py`](../code/visualizations/create_figures.py)
(Framework I outputs) and
[`create_figures_pnb.py`](../code/visualizations/create_figures_pnb.py)
(Framework II outputs). Both share the lesion-distribution panel (§6.1) and
the general rendering approach: parcellation-derived NIfTI volumes are
projected onto an `fsaverage5` cortical surface mesh and rendered from
axial, coronal and sagittal viewpoints (`utils.plot_block_surface`); block
graphs are rendered with `graph-tool`'s native hierarchy drawing
(`utils.plot_sbm_state`) rather than a surface projection, since block
adjacency is not itself a spatial quantity.

### 6.1 Sample lesion distribution

**Figure 1. Group-level lesion distribution.** Voxelwise lesion frequency
(`LesionAggregate.nii.gz`, patient-aggregate lesion overlap map) overlaid in
a plasma colourmap on the T1-weighted MNI152 template, shown across eight
axial slices (z = 40–145 mm, 15 mm nominal spacing). Colour intensity
indicates the proportion/count of patients with a lesion at that voxel;
non-lesioned voxels are transparent. Shared, unmodified input across both
frameworks. (`LesionDistribution.svg`)

### 6.2 Framework I figures (`create_figures.py`)

**Figure 2. Example disconnectome construction.** A single representative
patient's white-matter disconnectome, rendered as a semi-transparent glass
cortical surface (both hemispheres) with tractography streamlines
intersecting the lesion mask (grey, low opacity) and the lesion region of
interest itself rendered as a solid extracted isosurface (magenta, marching
cubes), shown from axial, coronal and sagittal viewpoints. Illustrates how
a single patient's binary disconnectome matrix (input to §3.1) is derived
from lesion-seeded tractography. (`disconnectome_example.svg`)

**Figure 3. Single-layer reference SBM fits per graph layer.** For each
task and each of the two graph layers independently (behaviour-weighted
co-occurrence; raw co-occurrence; §3.1, prior to joint modelling), the
corresponding thresholded adjacency matrix was fit with an independent,
non-layered, non-annealed nested SBM (`gt.minimize_nested_blockmodel_dl`,
no MCMC refinement) purely for illustrative comparison against the joint
model (Figure 5). Two panels per layer: nodes coloured by fitted block
membership (`tab20` categorical palette, one representative node per block
labelled with its block index, accompanying legend mapping block colour to
majority anatomical-group label among its members), and nodes coloured/
sized by degree with edges coloured by |edge weight| (log-normalised).
(`SBM_state_{task}_{behaviour,cooccurrence}_blocks.svg`,
`..._blocks_legend.svg`, `..._weights.svg`)

**Figure 4. Joint block z-score maps on cortical surface.** For each task
and each meaningful hierarchy level (0, 1) of the jointly fit layered SBM
(§3.2–3.3), the level's per-block z-scored behavioural-relevance value
(§3.5; consistency-weighted mean behaviour-layer degree, z-scored across
blocks) was mapped to every voxel of its member regions and projected onto
the cortical surface (plasma colourmap; axial/coronal/sagittal views).
Includes both z > 0 (selected/relevant) and z ≤ 0 blocks, so the full
range of block-level behavioural relevance is visible, not only the
selected subset. (`SBM_blocks_surface_{task}_layer{level}.svg`)

**Figure 5. Within-block mean edge weight, selected blocks.** For each task
and hierarchy level, and for each block selected as behaviourally relevant
(z-scored mean behaviour-layer degree ≥ 0, more than one member region;
§3.5), each member region's mean edge weight to other members of the same
block (normalised by the block's own maximum, not z-scored) was mapped to
the cortical surface (plasma colourmap, positive-only rendering; blocks
with no surviving intra-block edges under the final thresholded graph were
skipped). Complements Figure 4 by showing internal coupling strength within
each relevant block rather than the block's aggregate behavioural score.
(`SBM_block_edgeweights_{task}_lvl{level}_block{blk}.svg`)

**Figure 6. Final joint community structure.** For each task, the fully
converged joint nested SBM state (behaviour-weighted co-occurrence +
co-occurrence layers combined into one adjacency for display;
`load_joint_adjacency`, edge weight = sum of both layers' weights) is drawn
using the modal block assignments from every meaningful hierarchy level
(§3.4) as an explicit nested partition (rather than an independent re-fit),
so the displayed hierarchy is the one actually reported in the per-ROI
assignment tables. As in Figure 3, two panels: block-coloured with legend,
and degree-coloured/sized with |edge weight|-driven edge colour and alpha.
(`SBM_final_state_{task}_joint_blocks.svg`, `..._blocks_legend.svg`,
`..._weights.svg`)

### 6.3 Framework II figures (`create_figures_pnb.py`)

**Figure 7. Node-strength input matrices.** Subject × region matrices
(rows = subjects, columns = regions ordered left-to-right hemisphere) shown
as heatmaps (plasma colourmap, [0, 1]): the raw, globally rescaled node-
strength matrix common to all tasks (leftmost panel), and, for each of the
three tasks, the behaviour-weighted variant obtained by multiplying each
subject's rescaled row by that subject's own rescaled behavioural score
(§4.1) — the literal input ($S^T$) to the `beh_weighted`
`PseudoNormalBlockState` fit for that task. Zero entries are transparent.
(`NodeOccurrenceMatrix.svg`)

**Figure 8. Block behavioural-relevance maps on cortical surface.** For
each task, each input variant (`no_beh`, `beh_weighted`; §4.1) and each
hierarchy level (0, 1), every region's own block's mean region–behaviour
relevance value (§4.4; Spearman correlation with the raw behavioural score,
averaged within block) was mapped to the cortical surface (plasma
colourmap; axial/coronal/sagittal views), analogous to Figure 4 but scored
by direct behavioural correlation rather than consistency-weighted degree,
and reflecting the block-value NIfTI written by `run_recon_pnb.py`
(`{atlas}_{variant}_lvl{level}_blockvalues.nii.gz`).
(`SBM_blocks_surf_{task}_{variant}_layer{level}.svg`)

**Figure 9. Selected-block region membership.** For each task, variant and
hierarchy level, every block with z-scored mean relevance > 0.5 (§4.4) is
rendered as a flat, single-valued membership map (all member-region voxels
= 1, reverse-plasma colourmap, positive-only rendering) on the cortical
surface, one figure per selected block. Unlike an edge-weight map, pure
membership is shown because `PseudoNormalBlockState` groups regions by
similarity of coupling *profile* rather than by mutual direct connection, so
a genuinely relevant block can legitimately have few or no intra-block
edges — an edge-weight-based map would misleadingly render such blocks as
empty. (`SBM_block_members_{task}_{variant}_lvl{level}_block{blk}.svg`)

**Figure 10. Reconstructed network community structure.** For each task and
input variant, the fitted latent coupling graph (`load_pnb_adjacency`, edge
weight = fitted coupling strength $x$, which is frequently negative/
anti-correlated and is therefore handled by sign-agnostic edge presence and
|weight|-based colour/alpha scaling throughout) is drawn with its own
level-0 and level-1 block assignments as an explicit nested partition
(§4.3–4.4), in the same two-panel layout as Figure 6 (block-coloured +
legend; degree-coloured/sized), except that edge colour in the
degree-coloured panel is a genuine two-colour gradient between each edge's
endpoints' own region–behaviour relevance values (rather than a flat
average or raw-weight colouring), so this panel visualises how behavioural
relevance is distributed across the reconstructed coupling structure.
(`SBM_final_state_{task}_{variant}_blocks.svg`, `..._blocks_legend.svg`,
`..._weights.svg`)

---

## Notes for completion

- `[TODO]` markers above indicate content not derivable from `run.py` /
  `run_recon_pnb.py` / the two `create_figures*.py` scripts alone
  (participant recruitment/demographics, lesion segmentation and
  tractography acquisition parameters, task battery references, exact
  software versions) and must be filled in from the corresponding
  acquisition/preprocessing pipeline and package manifest.
- All bracketed `[Peixoto, ...]` citations should be checked against the
  installed `graph-tool` version's own bibliography
  (`graph_tool.inference` module docs) before submission — they are
  included here as best-effort placeholders for the general SBM,
  layered/edge-covariate SBM, merge-split MCMC, and pseudo-likelihood
  network reconstruction methodology actually invoked by the code, not
  verified against a live citation database.
- Figure numbering above follows script execution order within each file
  and is provisional; final figure order/inclusion should be set once the
  Results section is drafted, since not every generated `.svg` (e.g. the
  per-block panels of Figures 5, 9) is necessarily intended for the main
  text rather than supplementary material.
