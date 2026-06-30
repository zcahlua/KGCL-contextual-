from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kgcl_retro.models.contextual_fg import ContextualFGGraphEncoder


def _encoder():
    return ContextualFGGraphEncoder(
        atom_fdim=5,
        bond_fdim=3,
        hidden_size=7,
        fg_type_vocab_size=4,
        fg_max_dist=8,
        layers=1,
        dropout=0.0,
    )


def test_empty_core_mask_does_not_produce_nan_and_core_null_gets_grad():
    encoder = _encoder()
    graph_tensors = SimpleNamespace(
        fg_node_atom_features=torch.randn(3, 5),
        fg_node_core_mask=torch.tensor([False, False, False], dtype=torch.bool),
        fg_node_dist_to_core=torch.tensor([0, 1, 1], dtype=torch.long),
        fg_node_fg_type=torch.tensor([0, 1, 1], dtype=torch.long),
        fg_edge_index=torch.zeros((2, 0), dtype=torch.long),
        fg_edge_features=torch.zeros((0, 3)),
        fg_node_scope=[(1, 2)],
    )
    graph_tensors.fg_node_atom_features[0].zero_()

    out = encoder(graph_tensors)
    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert encoder.core_null.grad is not None


def test_empty_context_graph_returns_empty_instance_matrix():
    encoder = _encoder()
    graph_tensors = SimpleNamespace(
        fg_node_atom_features=torch.zeros((1, 5)),
        fg_node_core_mask=torch.tensor([False], dtype=torch.bool),
        fg_node_dist_to_core=torch.tensor([0], dtype=torch.long),
        fg_node_fg_type=torch.tensor([0], dtype=torch.long),
        fg_edge_index=torch.zeros((2, 0), dtype=torch.long),
        fg_edge_features=torch.zeros((0, 3)),
        fg_node_scope=[],
    )

    out = encoder(graph_tensors)

    assert out.shape == (0, 7)
