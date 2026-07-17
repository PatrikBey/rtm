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