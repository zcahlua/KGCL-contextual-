import pytest
import torch

pytest.importorskip("rdkit")

from rdkit import Chem

from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import get_batch_graphs
from kgcl_retro.models import KGCL


def _config(fg_mode):
    return {
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,
        "mpn_size": 32,
        "mlp_size": 64,
        "depth": 2,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": False,
        "use_attn": False,
        "n_heads": 4,
        "fg_mode": fg_mode,
        "fg_context_radius": 1,
        "fg_hidden_size": 32,
        "fg_layers": 2,
        "fg_dropout": 0.0,
        "fg_attn_dim": 16,
        "fg_max_dist": 8,
        "fg_max_matches_per_pattern": None,
        "fg_use_kg_fusion": True,
        "fg_use_membership_bias": True,
        "fg_use_distance_bias": True,
        "fg_null_token": True,
        "fg_freeze_kg_embeddings": False,
    }


def _vocabs():
    atom_vocab = Vocab([("Change Atom", (0, 0, 0, 0))])
    bond_vocab = Vocab([("Delete Bond", None)])
    return atom_vocab, bond_vocab


@pytest.mark.parametrize("fg_mode", ["legacy", "contextual", "none"])
def test_compute_edit_scores_action_dimension_is_unchanged(fg_mode):
    mol = Chem.MolFromSmiles("CC(=O)O")
    graph = MolGraph(mol, fg_mode=fg_mode, fg_context_radius=1)
    graph_tensors, scopes = get_batch_graphs([graph], fg_mode=fg_mode)
    atom_vocab, bond_vocab = _vocabs()
    model = KGCL(_config(fg_mode), atom_vocab, bond_vocab)

    edit_scores, _, _, _ = model.compute_edit_scores(graph_tensors, scopes)

    expected_dim = graph.num_bonds * len(bond_vocab) + graph.num_atoms * len(atom_vocab) + 1
    assert len(edit_scores) == 1
    assert edit_scores[0].shape == (expected_dim,)


def test_contextual_mode_backward_reaches_fg_encoder_parameters():
    mol = Chem.MolFromSmiles("CC(=O)O")
    graph = MolGraph(mol, fg_mode="contextual", fg_context_radius=1)
    graph_tensors, scopes = get_batch_graphs([graph], fg_mode="contextual")
    atom_vocab, bond_vocab = _vocabs()
    model = KGCL(_config("contextual"), atom_vocab, bond_vocab)

    edit_scores, _, _, _ = model.compute_edit_scores(graph_tensors, scopes)
    loss = edit_scores[0].sum()
    loss.backward()

    fg_grads = [
        param.grad
        for name, param in model.named_parameters()
        if name.startswith("contextual_fg_encoder") and param.requires_grad
    ]
    assert fg_grads
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in fg_grads)
