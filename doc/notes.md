# PROJECT NODES TO KEEP TRACK OF




## OPEN QUESTIONS:


### Reflexive process (RP):
*1* to what degree is the RP task dependent?


### Pathological impact
*2* how does pathological impact drive RP or deliberate process (DP)

Clinical populations — LATER has been used to characterise patients with Parkinson's disease, Alzheimer's, schizophrenia, and hemispatial neglect. Because shift and swivel have different neural interpretations, the model can distinguish whether a patient group is slower because their signal mean is lower (shift) or because their signal is noisier (swivel).

## GENERAL NOTES

When comparing conditions (e.g. different prior probabilities of a target):
- Shift: lines are parallel — same slope (sigma shared), different intercept (mu/k varies). The decision signal's variability is unchanged; only its mean shifts.
- Swivel: lines pivot around a common point — same k = mu/sigma shared, sigma varies. The signal-to-noise ratio is constant but the overall gain changes.


Appropriate of Go/No-Go designs?


## IMPLEMENTATION

# SBM

So the CSV values in SBM_block_connectivity_lvl0_Foreperiod_Long_tau.cs are best interpreted as: the posterior expected edge density between each pair of communities, where the expectation is taken over the distribution of partitions visited during the accumulation phase.


They are the modal partition — not the final state after 250 iterations, and not a simple average.
Specifically, modal_assignments[k] comes from PartitionModeState(b_history[k], converge=True).get_max(g), which finds the single partition that maximises the posterior probability across all 250 samples after resolving label switching. It's the most frequently visited assignment for each node, in a canonical labelling.
So concretely: if across 250 samples node 47 is assigned to block 2 in 210 of them and block 5 in the remaining 40, it ends up in block 2 in the modal partition. The block membership in the txt file reflects that majority vote, not where the chain happened to land at iteration 250.
This is meaningful because it means the block files represent the most probable community structure under the posterior, rather than an arbitrary endpoint of the chain. However, the caveat is that nodes with genuinely ambiguous membership (split roughly evenly across samples) still get hard-assigned to one block — which is exactly what node_consistency was quantifying. A node with consistency 0.55 is in its assigned block by only a slim majority and arguably shouldn't be treated the same as one with consistency 0.98.



Good point. The raw marginal probability ignores the baseline chance level entirely. With 20 blocks, random assignment gives 1/20 = 5% per block, so 40% in the modal block is actually very strong signal — but raw consistency doesn't show that.
The cleanest fix is a chance-corrected consistency, analogous to Cohen's Kappa for a single category:
adjusted = (observed - 1/n_blocks) / (1 - 1/n_blocks)

A node always in its modal block → 1.0
A node at pure chance level → 0.0
A node less consistent than chance → negative

So your example: observed=0.4, n_blocks=20, chance=0.05 → adjusted = (0.4−0.05)/(1−0.05) ≈ 0.368, which correctly reflects that this node is far more consistent than chance would predict.


################
# NETWORK RECONSTRUCTION NOTES

Yes, that's a coherent design, and it does dodge the circularity issue from before since the behaviour-only graph would be a genuinely separate signal from the lesion-cooccurrence data being reconstructed. Two things worth being precise about before you build it, though — one structural, one about what "soft" actually means.

Mechanically, this is valid
bstate only supplies the per-node block labels; it doesn't carry over the edges it was originally fit on. The reconstruction's own graph comes from a separate g= argument. So fitting an SBM on a completely different edge set (your behaviour-only graph) and handing its partition to PseudoIsingBlockState(s, bstate=that_partition) for the lesion-only data is structurally fine — same node set, different edges, no conflict. The one gotcha: whatever you fit needs to match the nested= setting the reconstruction uses (default nested=True expects a NestedBlockState, not a flat BlockState), so fit it the same way.

The important correction: "finite sbm probability" isn't a prior strength, it's a mixing-rate throttle
This is the part I'd flag before you build it. sbm controls how often sbm_mcmc_sweep gets proposed, not how strongly each proposal is biased toward your initialization. Each time it does fire, it evaluates the standard, unweighted SBM move criterion — nothing about it "remembers" or is penalized for departing from the behaviour-derived partition. So in the limit of a long enough chain, sbm=0.01 and sbm=1 converge to the same stationary posterior — a low rate just means it takes longer to get there. What you'd actually be observing with a finite/moderate run isn't a soft prior in the Bayesian sense (a belief that persists in the final answer with some defined strength); it's the partition not having had enough moves yet to fully re-equilibrate away from where it started. That's a real, legitimate, commonly-used practical technique (it's the same logic behind warm-restarts and annealing schedules), but it's a runtime/compute-budget effect, not a formal regularization term — worth not overselling it as "the prior has weight X."

If you want a soft prior with an actual, well-defined strength — something where the final posterior itself (not just the convergence speed) reflects "trust the behaviour partition somewhat, but let strong lesion evidence override it" — you'd need the block-move acceptance criterion itself to include a term rewarding agreement with the behaviour partition (e.g. via entropy_args or a custom penalty), not just throttle how often moves are attempted. I haven't verified graph-tool exposes that kind of term directly for these reconstruction states — would need to check.

Practical recommendation
Given that distinction, I'd frame it as two honest options rather than a continuous dial:

sbm=0 — genuine hard constraint: the behaviour partition is the regularizer, full stop, and it will persist in the final answer regardless of runtime.
Small nonzero sbm, finite/bounded run — a legitimate "give it a head start and don't run so long that it forgets" approach, but its effect is coupled to how many sweeps you run — if you increase niter later without changing sbm, you should expect the behavioural influence to fade, which could be a surprising/hard-to-reproduce gotcha in practice.
I'd lean toward starting with sbm=0 (freeze it, see if the lesion-only reconstruction under that regularizer looks sane), then relaxing to a small nonzero value with a fixed, documented iteration budget if you want the lesion data to have some ability to push back — rather than treating "somewhere between 0 and 1" as a smoothly interpretable knob.


# behaviour integration into SBM block fit
Your description is close, with one refinement: it's not strictly "fit SBM, then derive edges" as two sequential stages per step — the block partition and the edge/weight parameters are updated jointly within mcmc_sweep(), each conditioned on the other's current state. But functionally that's exactly what you're pointing at: the nested SBM acts as a structured prior over which edges are plausible (edge probability depends on which blocks the two endpoints belong to), so the block structure shapes what gets proposed/accepted as an edge, while the fit to the observed data (S) shapes what partition best explains the resulting connectivity. They co-evolve. So "block structure drives the edge prior, edges get re-summarized into block structure" is the right mental model.

Your proposal is sound in principle — and it dodges a problem

Routing behaviour through the block prior rather than injecting it as a literal reconstructed node is actually a better fit for your Ising concern than my earlier answer: it sidesteps the binary-state-space mismatch entirely. Behaviour never needs to be binarized or treated as a spin, because it wouldn't be part of the Ising pseudo-likelihood at all — it would only bias which nodes get grouped together during the block-inference step. That's a real improvement over forcing behaviour into the same state space as brain nodes.

Where it runs into graph_tool's actual API boundaries

graph_tool does have the machinery for "multiple layers/covariates informing one shared partition" — that's gt.LayeredBlockState (and edge-covariate SBMs more generally, from Peixoto's "Inferring the mesoscale structure of layered, edge-valued, and time-varying networks"). It's built exactly for the case of a primary relation plus auxiliary layers, with a single block partition fit jointly across all of them.

The problem is that this machinery is designed for fully observed layers used for community detection — not for the case where one layer (brain connectivity) is itself latent and being reconstructed via pseudolikelihood from node-level data, which is what PseudoIsingBlockState/PseudoNormalBlockState do. As far as the documented API goes, those reconstruction classes don't expose a layers= or covariate_layer= hook to swap in a LayeredBlockState as their internal prior — internally they hold a plain NestedBlockState, and mcmc_sweep() updates both edge parameters and that internal partition together as one coupled process. There's no supported seam for "use this externally-fit multilayer partition as the prior for the next sweep."

So what you're describing is implementable, but only by hand-rolling an outer loop yourself rather than flipping a parameter:
1. Run a batch of reconstruction sweeps to get a current edge-weight estimate (as now).
2. Take that estimate, pair it as one layer of a LayeredBlockState alongside a fixed behaviour-derived layer, and fit that multilayer SBM to get a partition.
3. Feed that partition back into the reconstruction state (state.get_block_state() does return the live NestedBlockState, and it does accept an explicit b assignment) — but you'd also need to lock partition moves off in the reconstruction's own MCMC (so it doesn't just re-derive its own partition and overwrite yours next sweep) and let it only move edge parameters under that fixed grouping for a while.
4. Alternate steps 1–3.

That's a real engineering undertaking — a custom EM-style coupling between two separately-managed state objects — not a documented feature, and there's no guarantee the reconstruction's convergence proofs/pseudolikelihood consistency still hold once you're periodically overriding its partition from outside.

A design gap worth resolving before any of this

A "layer" in graph_tool is itself a graph — edges among the same node set. Your brain-connectivity layer is naturally that (nodes = regions, edges = inferred coupling). But "behaviour" as you have it is one scalar per subject, i.e., a property of observations, not a relation between regions. To make it a layer you'd need to first derive a region-by-region matrix from behaviour — e.g., an outer-product-style layer where edge (i,j) weight reflects how similarly regions i and j each relate to behaviour, or some other dyadic transform. That derivation is a modeling decision on your part; graph_tool won't produce it from the raw behaviour vector.

If instead what you actually want is simpler than a full layer — e.g., "each region has one scalar indicating its behavioural relevance, and blocks should be influenced by that" — that's a vertex covariate, not an edge layer, and graph_tool's plain BlockState does support partition-constraint/label covariates (pclabel) for exactly this without requiring you to invent a dyadic behaviour graph. That path is much less custom-engineering-heavy than the layered approach and might get you most of what you want (behaviour-informed block structure) more cheaply.

The tradeoff to weigh either way

This approach will never give you explicit brain-behaviour edges — only blocks that are behaviourally coherent. If the deliverable you actually need is "edge weights between behaviour and specific regions" (as your original node-injection design produced), this indirect route doesn't produce that object; you'd have to define some derived statistic afterward (e.g., which regions share a block with high behaviour-association) rather than reading off a weighted edge. Worth deciding now whether block-level behavioural coherence is actually sufficient for your downstream analysis, or whether you need literal brain-behaviour edge weights — that determines whether this is worth the custom-orchestration cost versus the earlier binarize-and-append approach.