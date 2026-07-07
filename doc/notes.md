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


