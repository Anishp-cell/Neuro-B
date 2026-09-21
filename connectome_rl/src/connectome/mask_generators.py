"""Mask Generators and Degree-Preserving Null Model Suite for Drosophila Connectomics.

Directly implements the null model requirements for the Digital Sphinx adversarial audit:
1. Exact preservation of in-degree and out-degree distributions for all descending neurons (DNs),
   interneurons (INs), and motor neurons (MNs).
2. Randomization of synaptic target specificity via bipartite double-edge swaps.
3. Generation of the 20-seed rewired null suite (seeds 101-120) with shared initial policy weights.
"""

from __future__ import annotations

import copy
import logging
from typing import Sequence

import numpy as np
import torch

from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData

logger = logging.getLogger(__name__)


def bipartite_double_edge_swap(
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_swaps: int = 5000,
    max_tries_factor: int = 5,
    seed: int = 42,
) -> tuple[torch.BoolTensor, torch.FloatTensor | None]:
    """Perform degree-preserving double-edge swaps on a bipartite adjacency mask.

    For a bipartite layer Y = W @ X:
      - mask shape: (R, C) where R is target nodes (rows), C is source nodes (cols).
      - In-degree of target node v is row_sum(v).
      - Out-degree of source node u is col_sum(u).

    Each valid swap selects edges (r1, c1) and (r2, c2) with r1 != r2 and c1 != c2,
    and replaces them with (r1, c2) and (r2, c1) if neither edge exists.
    This guarantees mathematically that row sums and column sums are EXACTLY preserved.

    Args:
        mask: Boolean tensor of shape (R, C).
        weights: Optional synaptic weight tensor of shape (R, C).
        n_swaps: Target number of successful edge swaps to perform.
        max_tries_factor: Multiplier for maximum attempts (max_tries = n_swaps * factor).
        seed: Random seed for swap sampling.

    Returns:
        tuple of (new_mask, new_weights)
    """
    rng = np.random.RandomState(seed)
    m = mask.clone().cpu().numpy().astype(bool)
    w = weights.clone().cpu().numpy() if weights is not None else None

    rows, cols = np.where(m)
    num_edges = len(rows)
    if num_edges < 2:
        return mask.clone(), (weights.clone() if weights is not None else None)

    max_tries = n_swaps * max_tries_factor
    successful_swaps = 0

    for _ in range(max_tries):
        if successful_swaps >= n_swaps:
            break

        i1, i2 = rng.choice(num_edges, size=2, replace=False)
        r1, c1 = rows[i1], cols[i1]
        r2, c2 = rows[i2], cols[i2]

        if r1 != r2 and c1 != c2 and not m[r1, c2] and not m[r2, c1]:
            # Apply swap
            m[r1, c1] = False
            m[r2, c2] = False
            m[r1, c2] = True
            m[r2, c1] = True

            rows[i1], cols[i1] = r1, c2
            rows[i2], cols[i2] = r2, c1

            if w is not None:
                w1 = w[r1, c1]
                w2 = w[r2, c2]
                w[r1, c1] = 0.0
                w[r2, c2] = 0.0
                w[r1, c2] = w1
                w[r2, c1] = w2

            successful_swaps += 1

    new_mask = torch.from_numpy(m).bool()
    new_w = torch.from_numpy(w).float() if w is not None else None

    logger.debug(
        f"Completed {successful_swaps}/{n_swaps} bipartite double-edge swaps (seed={seed})."
    )
    return new_mask, new_w


def generate_degree_matched_null(
    circuit_data: ConnectomeCircuitData,
    seed: int,
    n_swaps_hop1: int = 1500,
    n_swaps_hop2: int = 6000,
) -> ConnectomeCircuitData:
    """Generate a degree-preserving rewired null model from biological ConnectomeCircuitData.

    Preserves exact in-degree and out-degree distributions for all DNs, INs, and MNs
    while shuffling synaptic target specificity.

    Args:
        circuit_data: Original biological ConnectomeCircuitData instance.
        seed: Random seed for rewiring.
        n_swaps_hop1: Number of double-edge swaps on Hop 1 (DN -> IN).
        n_swaps_hop2: Number of double-edge swaps on Hop 2 (IN -> MN).

    Returns:
        New ConnectomeCircuitData instance with rewired masks and identical node mappings.
    """
    # 1. Swap Hop 1: DN -> IN
    new_hop1_mask, new_hop1_weights = bipartite_double_edge_swap(
        mask=circuit_data.hop1_mask,
        weights=circuit_data.hop1_weights,
        n_swaps=n_swaps_hop1,
        seed=seed,
    )

    # 2. Swap Hop 2: IN -> MN (use distinct seed offset to avoid correlation)
    new_hop2_mask, new_hop2_weights = bipartite_double_edge_swap(
        mask=circuit_data.hop2_mask,
        weights=circuit_data.hop2_weights,
        n_swaps=n_swaps_hop2,
        seed=seed + 10000,
    )

    # 3. Reconstruct full (N, N) adjacency mask & weights
    num_nodes = circuit_data.num_nodes
    new_full_mask = circuit_data.full_mask.clone()
    new_full_weights = circuit_data.full_weights.clone()

    dn_globals = circuit_data.dn_indices
    in_globals = circuit_data.in_indices
    mn_globals = circuit_data.mn_indices

    # Clear old Hop 1 & Hop 2 in full_mask
    for c_idx, dn_g in enumerate(dn_globals):
        for r_idx, in_g in enumerate(in_globals):
            new_full_mask[in_g, dn_g] = False
            new_full_weights[in_g, dn_g] = 0.0

    for c_idx, in_g in enumerate(in_globals):
        for r_idx, mn_g in enumerate(mn_globals):
            new_full_mask[mn_g, in_g] = False
            new_full_weights[mn_g, in_g] = 0.0

    # Insert rewired Hop 1
    r_idx, c_idx = torch.where(new_hop1_mask)
    for r, c in zip(r_idx.tolist(), c_idx.tolist()):
        tgt_g = in_globals[r]
        src_g = dn_globals[c]
        new_full_mask[tgt_g, src_g] = True
        if new_hop1_weights is not None:
            new_full_weights[tgt_g, src_g] = new_hop1_weights[r, c]

    # Insert rewired Hop 2
    r_idx, c_idx = torch.where(new_hop2_mask)
    for r, c in zip(r_idx.tolist(), c_idx.tolist()):
        tgt_g = mn_globals[r]
        src_g = in_globals[c]
        new_full_mask[tgt_g, src_g] = True
        if new_hop2_weights is not None:
            new_full_weights[tgt_g, src_g] = new_hop2_weights[r, c]

    # 4. Reconstruct PyG edge_index and edge_weight
    tgt_list, src_list = torch.where(new_full_mask)
    edge_index = torch.stack([src_list, tgt_list], dim=0)  # source -> target
    edge_weight = new_full_weights[tgt_list, src_list]

    return ConnectomeCircuitData(
        num_nodes=num_nodes,
        node_to_idx=copy.deepcopy(circuit_data.node_to_idx),
        idx_to_node=copy.deepcopy(circuit_data.idx_to_node),
        dn_indices=list(circuit_data.dn_indices),
        in_indices=list(circuit_data.in_indices),
        mn_indices=list(circuit_data.mn_indices),
        full_mask=new_full_mask,
        full_weights=new_full_weights,
        hop1_mask=new_hop1_mask,
        hop1_weights=new_hop1_weights if new_hop1_weights is not None else circuit_data.hop1_weights.clone(),
        hop2_mask=new_hop2_mask,
        hop2_weights=new_hop2_weights if new_hop2_weights is not None else circuit_data.hop2_weights.clone(),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def generate_null_suite(
    circuit_data: ConnectomeCircuitData,
    num_models: int = 20,
    base_seed: int = 101,
) -> list[ConnectomeCircuitData]:
    """Generate the full ensemble of degree-preserving rewired null models (e.g. 20 seeds).

    Args:
        circuit_data: Biological ConnectomeCircuitData.
        num_models: Number of null models to generate (default: 20).
        base_seed: Starting seed integer (default: 101 -> seeds 101 to 120).

    Returns:
        List of 20 distinct rewired ConnectomeCircuitData models.
    """
    null_suite: list[ConnectomeCircuitData] = []
    for i in range(num_models):
        seed = base_seed + i
        null_model = generate_degree_matched_null(circuit_data=circuit_data, seed=seed)
        null_suite.append(null_model)
    return null_suite


def verify_degree_conservation(
    orig: ConnectomeCircuitData,
    null_model: ConnectomeCircuitData,
) -> dict[str, float]:
    """Verify that in-degree and out-degree distributions are 100% mathematically conserved.

    Returns:
        Dictionary reporting max absolute differences for in-degree and out-degree.
    """
    hop1_in_diff = float((orig.hop1_mask.sum(dim=1) - null_model.hop1_mask.sum(dim=1)).abs().max())
    hop1_out_diff = float((orig.hop1_mask.sum(dim=0) - null_model.hop1_mask.sum(dim=0)).abs().max())

    hop2_in_diff = float((orig.hop2_mask.sum(dim=1) - null_model.hop2_mask.sum(dim=1)).abs().max())
    hop2_out_diff = float((orig.hop2_mask.sum(dim=0) - null_model.hop2_mask.sum(dim=0)).abs().max())

    hop1_overlap = float((orig.hop1_mask & null_model.hop1_mask).sum()) / float(orig.hop1_mask.sum())
    hop2_overlap = float((orig.hop2_mask & null_model.hop2_mask).sum()) / float(orig.hop2_mask.sum())

    return {
        "hop1_max_in_degree_diff": hop1_in_diff,
        "hop1_max_out_degree_diff": hop1_out_diff,
        "hop2_max_in_degree_diff": hop2_in_diff,
        "hop2_max_out_degree_diff": hop2_out_diff,
        "hop1_edge_overlap": hop1_overlap,
        "hop2_edge_overlap": hop2_overlap,
    }
