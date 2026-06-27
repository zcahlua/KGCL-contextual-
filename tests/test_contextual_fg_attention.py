from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kgcl_retro.models.contextual_fg import AtomFGAttention


def _graph_tensors():
    return SimpleNamespace(
        fg_scope=[(0, 2), (2, 0)],
        atom_fg_membership=torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [False, False],
            ],
            dtype=torch.bool,
        ),
        atom_fg_dist=torch.tensor(
            [
                [9, 9],
                [0, 2],
                [2, 0],
                [9, 9],
            ],
            dtype=torch.long,
        ),
    )


def test_attention_handles_two_molecules_and_zero_fg_molecule():
    attention = AtomFGAttention(atom_fdim=5, fg_dim=7, attn_dim=3, fg_max_dist=8, use_null_token=True)
    atom_features = torch.randn(4, 5, requires_grad=True)
    atom_features.data[0].zero_()
    fg_embeddings = torch.randn(2, 7, requires_grad=True)

    enhanced, fg_context, attention_weights = attention(
        atom_features,
        fg_embeddings,
        _graph_tensors(),
        atom_scope=[(1, 2), (3, 1)],
    )

    assert enhanced.shape == atom_features.shape
    assert fg_context.shape == (4, 3)
    assert len(attention_weights) == 2
    assert attention_weights[0].shape == (2, 3)
    assert attention_weights[1].shape == (1, 1)
    assert enhanced[0].abs().sum() == 0
    assert fg_context[0].abs().sum() == 0


def test_attention_uses_null_token_when_null_flag_false_and_no_real_fgs():
    attention = AtomFGAttention(atom_fdim=5, fg_dim=7, attn_dim=3, fg_max_dist=8, use_null_token=False)
    atom_features = torch.randn(2, 5)
    atom_features[0].zero_()
    graph_tensors = SimpleNamespace(
        fg_scope=[(0, 0)],
        atom_fg_membership=torch.zeros((2, 0), dtype=torch.bool),
        atom_fg_dist=torch.zeros((2, 0), dtype=torch.long),
    )

    enhanced, fg_context, attention_weights = attention(
        atom_features,
        torch.zeros((0, 7)),
        graph_tensors,
        atom_scope=[(1, 1)],
    )

    assert enhanced.shape == atom_features.shape
    assert fg_context.shape == (2, 3)
    assert attention_weights[0].shape == (1, 1)


def test_attention_backward_reaches_projection_parameters():
    attention = AtomFGAttention(atom_fdim=5, fg_dim=7, attn_dim=3, fg_max_dist=8, use_null_token=True)
    atom_features = torch.randn(4, 5, requires_grad=True)
    atom_features.data[0].zero_()
    fg_embeddings = torch.randn(2, 7, requires_grad=True)

    enhanced, fg_context, _ = attention(atom_features, fg_embeddings, _graph_tensors(), atom_scope=[(1, 2), (3, 1)])
    loss = enhanced.sum() + fg_context.sum()
    loss.backward()

    for layer in (attention.query, attention.key, attention.value, attention.output):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
