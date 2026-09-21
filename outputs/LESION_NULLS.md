# The Digital Sphinx In-Silico Lesion Audit: Biological Specificity vs. 20 Degree-Matched Rewired Controls

**Target Outreach Briefing for Dr. Bingni Brunton, Elliott Abe (UW), Dr. John Tuthill (UW), and Dr. Srinivas Turaga (HHMI Janelia).**

---

## Executive Summary

In *The digital sphinx: Can a worm brain control a fly body?* (bioRxiv, doi:10.64898/2026.03.20.713233), Brunton, Abe, Hu, and Tuthill established that unconstrained deep reinforcement learning can induce locomotion in arbitrary artificial network architectures (including a nematode worm connectome in a fly body), warning that high-level behavioral replication alone is insufficient to prove the biological realism of connectome-constrained policies.

Here, we report the results of the **Pre-Registered Adversarial Lesion Audit** on the *Drosophila melanogaster* Janelia MaleCNS v1.0 descending motor connectome. We test whether the stereotyped asymmetric steering yaw bias induced by targeted unilateral ablation of descending neuron **DNa01** reflects true biological wiring specificity or can be reproduced by degree-and-synapse-matched random graph null models under identical embodied physics.

To address training instability concerns in deep reinforcement learning (*Henderson et al., 2018*), we evaluate an ensemble of **N = 10 independently-seeded biological policy instances** (seeds 42 to 51) against **20 degree-matched rewired null models** (seeds 101 to 120; 60 paired rollouts).

---

## The Pre-Registered Falsification Verdict

> [!IMPORTANT]
> **PRE-REGISTERED FALSIFICATION RULE (Declared in Advance):**
> - **Hypothesis Confirmed IF**: Biological DNa01 unilateral ablation produces a lateralized turning yaw rate $> 15^\circ/\text{s}$, its 95% bootstrap confidence interval does NOT overlap with the 20 degree-matched rewired null models, AND $< 10$ of 20 rewired nulls reproduce turning $> 15^\circ/\text{s}$.
> - **Hypothesis Falsified IF**: The confidence intervals overlap, OR $\ge 10$ of 20 rewired nulls reproduce turning $> 15^\circ/\text{s}$.

### **Audit Outcome: HYPOTHESIS CONFIRMED: Biological connectome possesses motor functional specificity that degree-matched random networks fail to replicate under unilateral ablation.**

| Preregistered Criterion | Empirical Result | Status |
| :--- | :--- | :--- |
| **1. Biological Yaw Rate $> 15.0^\circ/\text{s}$** | **38.19°/s** (95% CI: [32.37, 45.22]°/s) | **PASS** |
| **2. Non-Overlapping 95% Bootstrap CIs** | Biological: [32.37, 45.22] vs Null: [2.12, 3.79]°/s | **PASS** |
| **3. $< 10$ Rewired Nulls Replicating Turn** | **0 of 20** rewired nulls exceeded $15^\circ/\text{s}$ | **PASS** |

---

## Quantitative Comparison Table

| Model Architecture | N (Models / Seeds) | Start-up Speed (mm/s) | Steady-State Speed (mm/s) | Induced Turn Rate (°/s) | 95% Bootstrap CI (°/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Biological MaleCNS v1.0** | **10** | **+7.9** | **+12.7** (5.1 BL/s) | **38.19°/s** | **[32.37, 45.22]** |
| **Rewired Null Ensemble (20 Models)** | **60 (20 models × 3 seeds)** | **+8.4** | **+12.7** (5.1 BL/s) | **2.92°/s** | **[2.12, 3.79]** |

### Individual Rewired Null Models Summary (Seeds 101 to 120):
- **Exact Degree Conservation**: 0.0 maximum deviation across all 506 neurons (Hop 1 and Hop 2 in/out degree error = 0.0).
- **Average Hop 2 Synaptic Overlap**: 7.0%.
- **Models exceeding $15^\circ/\text{s}$ turning**: 0 / 20.

---

## Neurobiological Implications

1. **Functional Specificity Established Across Independent Training Runs**: Unilateral ablation of DNa01 across **all 10 independent biological models** produces a consistent, stereotyped steering bias of **38.19°/s** (95% CI: [32.37, 45.22]°/s), aligning with in-vivo optogenetic recordings (*Namiki et al., 2018; Rayshubskiy et al., 2020, 2025*).
2. **Degree-Preserving Nulls Fail to Steer**: When identical synapse counts and exact degree distributions are rewired across VNC interneurons, single-neuron ablation fails to elicit coherent steering (**2.92°/s**), producing diffuse bilateral slowing or uncoordinated stumbling.
3. **Resolution of the Digital Sphinx Challenge**: While DRL can train arbitrary neural architectures to produce forward locomotion, **only genuine biological connectome wiring preserves single-cell lesion vulnerability and functional motor localization**.

---

*Author: Anish Pathak (Independent Researcher) | Repository: https://github.com/Anishp-cell/connectome-drl*
