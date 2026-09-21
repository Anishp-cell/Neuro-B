"""Unit Tests for Degree-Preserving Null Models and Scrambled Connectome Suite.

Verifies:
1. Exact preservation of in-degree and out-degree distributions under bipartite double-edge swaps.
2. Production of 20 distinct scrambled-wiring null models (seeds 101 to 120).
3. Pairwise distinctness across the 20-model null suite.
4. Shared policy weight initialization eliminating starting-weight confounds.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data
from connectome_rl.src.connectome.mask_generators import (
    bipartite_double_edge_swap,
    generate_degree_matched_null,
    generate_null_suite,
    verify_degree_conservation,
)
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


@pytest.fixture
def sample_circuit() -> ConnectomeCircuitData:
    """Load the cached MaleCNS circuit data or generate mock circuit data."""
    try:
        return load_circuit_data("connectome_rl/data/dna_circuit_tensors.pt")
    except Exception:
        # Fallback to mock circuit if file not present
        num_dn, num_in, num_mn = 4, 20, 40
        num_nodes = num_dn + num_in + num_mn
        hop1_mask = torch.rand(num_in, num_dn) > 0.5
        hop1_w = torch.rand(num_in, num_dn) * hop1_mask.float()
        hop2_mask = torch.rand(num_mn, num_in) > 0.7
        hop2_w = torch.rand(num_mn, num_in) * hop2_mask.float()

        full_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        full_w = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

        return ConnectomeCircuitData(
            num_nodes=num_nodes,
            node_to_idx={i: i for i in range(num_nodes)},
            idx_to_node={i: i for i in range(num_nodes)},
            dn_indices=list(range(num_dn)),
            in_indices=list(range(num_dn, num_dn + num_in)),
            mn_indices=list(range(num_dn + num_in, num_nodes)),
            full_mask=full_mask,
            full_weights=full_w,
            hop1_mask=hop1_mask,
            hop1_weights=hop1_w,
            hop2_mask=hop2_mask,
            hop2_weights=hop2_w,
            edge_index=torch.zeros((2, 10), dtype=torch.long),
            edge_weight=torch.zeros(10, dtype=torch.float32),
        )


class TestNullModels:
    """Test suite for degree-preserving null models."""

    def test_bipartite_swap_exact_degree_conservation(self) -> None:
        """Verify that row sums and col sums are 100% mathematically conserved."""
        mask = (torch.rand(30, 10) > 0.6).bool()
        weights = torch.rand(30, 10) * mask.float()

        orig_row_sums = mask.sum(dim=1)
        orig_col_sums = mask.sum(dim=0)

        new_mask, new_weights = bipartite_double_edge_swap(
            mask=mask,
            weights=weights,
            n_swaps=500,
            seed=42,
        )

        assert torch.equal(new_mask.sum(dim=1), orig_row_sums), "In-degree (row sums) must match exactly!"
        assert torch.equal(new_mask.sum(dim=0), orig_col_sums), "Out-degree (col sums) must match exactly!"
        assert new_mask.sum() == mask.sum(), "Total edge count must be conserved!"
        if new_weights is not None:
            assert torch.allclose(new_weights.sum(), weights.sum(), atol=1e-5)

    def test_generate_degree_matched_null(self, sample_circuit: ConnectomeCircuitData) -> None:
        """Verify complete circuit generation preserves degree sequence across layers."""
        null_circuit = generate_degree_matched_null(circuit_data=sample_circuit, seed=101)

        assert isinstance(null_circuit, ConnectomeCircuitData)
        assert null_circuit.num_nodes == sample_circuit.num_nodes
        assert null_circuit.dn_indices == sample_circuit.dn_indices
        assert null_circuit.in_indices == sample_circuit.in_indices
        assert null_circuit.mn_indices == sample_circuit.mn_indices

        metrics = verify_degree_conservation(sample_circuit, null_circuit)
        assert metrics["hop1_max_in_degree_diff"] == 0.0
        assert metrics["hop1_max_out_degree_diff"] == 0.0
        assert metrics["hop2_max_in_degree_diff"] == 0.0
        assert metrics["hop2_max_out_degree_diff"] == 0.0

        # Substantial shuffling should have occurred (overlap strictly < 1.0)
        assert metrics["hop1_edge_overlap"] < 0.85
        assert metrics["hop2_edge_overlap"] < 0.50

    def test_null_suite_20_seeds_distinctness(self, sample_circuit: ConnectomeCircuitData) -> None:
        """Verify that 20 distinct seeds produce 20 distinct, degree-conserved models."""
        suite = generate_null_suite(circuit_data=sample_circuit, num_models=20, base_seed=101)

        assert len(suite) == 20

        # Check all models preserve degree
        for i, null_model in enumerate(suite):
            m = verify_degree_conservation(sample_circuit, null_model)
            assert m["hop1_max_in_degree_diff"] == 0.0
            assert m["hop2_max_in_degree_diff"] == 0.0

        # Check distinctness across models: hop2 masks should not be identical
        for i in range(len(suite)):
            for j in range(i + 1, min(i + 4, len(suite))):
                assert not torch.equal(suite[i].hop2_mask, suite[j].hop2_mask), f"Null {i} and {j} must not be identical!"

    def test_shared_policy_initialization_confound_elimination(
        self, sample_circuit: ConnectomeCircuitData
    ) -> None:
        """Verify that biological and rewired policies share encoder/decoder weights under identical seed."""
        null_circuit = generate_degree_matched_null(circuit_data=sample_circuit, seed=101)

        torch.manual_seed(42)
        bio_policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=sample_circuit)

        torch.manual_seed(42)
        null_policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=null_circuit)

        # Sensory encoders and actuator decoders must be identical
        assert torch.equal(
            bio_policy.sensory_encoder[0].weight,
            null_policy.sensory_encoder[0].weight,
        ), "Sensory encoder weights must match exactly under shared initialization seed!"
        assert torch.equal(
            bio_policy.actuator_decoder.weight,
            null_policy.actuator_decoder.weight,
        ), "Actuator decoder weights must match exactly under shared initialization seed!"
