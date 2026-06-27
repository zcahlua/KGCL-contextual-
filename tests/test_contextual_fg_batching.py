import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from kgcl_retro.chemistry.features import BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph
from kgcl_retro.data.collate import BatchGraphTensors, get_batch_graphs


def test_contextual_batch_tensors_preserve_scopes_and_add_fg_metadata():
    with_fg = MolGraph(Chem.MolFromSmiles("CC(=O)O"), fg_mode="contextual", fg_context_radius=1)
    no_fg = MolGraph(Chem.MolFromSmiles("[He]"), fg_mode="contextual", fg_context_radius=1)

    graph_tensors, scopes = get_batch_graphs([with_fg, no_fg], fg_mode="contextual")
    atom_scope, bond_scope = scopes

    assert isinstance(graph_tensors, BatchGraphTensors)
    assert atom_scope == [(1, with_fg.n_atoms), (1 + with_fg.n_atoms, no_fg.n_atoms)]
    assert bond_scope == [(1, with_fg.num_bonds), (1 + with_fg.num_bonds, no_fg.num_bonds)]
    assert graph_tensors.f_bond_attrs.shape == (1 + with_fg.n_bonds + no_fg.n_bonds, BOND_FDIM)
    assert graph_tensors.has_contextual_fg
    assert graph_tensors.fg_node_atom_features.size(0) >= graph_tensors.fg_node_core_mask.size(0)
    assert graph_tensors.fg_kg_embeddings.size(0) == len(with_fg.fg_instances)
    assert graph_tensors.fg_scope[0][1] == len(with_fg.fg_instances)
    assert graph_tensors.fg_scope[1][1] == 0


def test_contextual_atom_fg_metadata_does_not_cross_molecules():
    with_fg = MolGraph(Chem.MolFromSmiles("CC(=O)O"), fg_mode="contextual", fg_context_radius=1)
    no_fg = MolGraph(Chem.MolFromSmiles("[He]"), fg_mode="contextual", fg_context_radius=1)

    graph_tensors, scopes = get_batch_graphs([with_fg, no_fg], fg_mode="contextual")
    atom_scope, _ = scopes
    second_atom_start, second_atom_count = atom_scope[1]
    first_fg_start, first_fg_count = graph_tensors.fg_scope[0]
    second_atom_slice = slice(second_atom_start, second_atom_start + second_atom_count)
    first_fg_slice = slice(first_fg_start, first_fg_start + first_fg_count)

    assert not graph_tensors.atom_fg_membership[second_atom_slice, first_fg_slice].any()
    assert (
        graph_tensors.atom_fg_dist[second_atom_slice, first_fg_slice]
        == graph_tensors.fg_max_dist + 1
    ).all()


def test_none_mode_skips_legacy_attention_and_contextual_metadata():
    mol = Chem.MolFromSmiles("CC(=O)O")
    graph = MolGraph(mol, fg_mode="none")

    assert graph.f_atoms == graph.f_atoms_raw
    assert graph.f_fgs == []
    assert graph.fg_instances == []
