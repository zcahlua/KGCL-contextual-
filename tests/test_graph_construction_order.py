import pytest

pytest.importorskip("rdkit")

import torch
from rdkit import Chem

from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import get_batch_graphs, prepare_edit_labels


def _manual_mol_with_non_pairscan_bond_indices():
    mol = Chem.RWMol()
    for map_num, symbol in enumerate(["C", "C", "O", "N"], start=1):
        atom = Chem.Atom(symbol)
        atom.SetAtomMapNum(map_num)
        mol.AddAtom(atom)
    mol.AddBond(2, 3, Chem.BondType.SINGLE)
    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol.AddBond(1, 2, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _old_pair_scan_directed_pairs(mol):
    pairs = []
    for a1 in range(mol.GetNumAtoms()):
        for a2 in range(a1 + 1, mol.GetNumAtoms()):
            if mol.GetBondBetweenAtoms(a1, a2) is not None:
                pairs.append((a1, a2))
                pairs.append((a2, a1))
    return pairs


def _graph_directed_pairs(graph):
    dest_by_bond = {}
    for atom_idx, incoming_bonds in enumerate(graph.a2b):
        for bond_idx in incoming_bonds:
            dest_by_bond[bond_idx] = atom_idx
    return [(graph.b2a[bond_idx], dest_by_bond[bond_idx]) for bond_idx in range(graph.n_bonds)]


def test_bond_iteration_matches_old_pair_scan_directed_order():
    mol = _manual_mol_with_non_pairscan_bond_indices()

    graph = MolGraph(mol, fg_mode="none")

    assert _graph_directed_pairs(graph) == _old_pair_scan_directed_pairs(mol)


def test_a2b_b2a_b2revb_consistent_after_bond_iteration_optimization():
    mol = _manual_mol_with_non_pairscan_bond_indices()

    graph = MolGraph(mol, fg_mode="none")

    directed_pairs = _graph_directed_pairs(graph)
    for bond_idx, reverse_idx in enumerate(graph.b2revb):
        assert graph.b2revb[reverse_idx] == bond_idx
        assert directed_pairs[reverse_idx] == tuple(reversed(directed_pairs[bond_idx]))


def test_undirected_b2a_stays_aligned_with_rdkit_bond_indices():
    mol = _manual_mol_with_non_pairscan_bond_indices()
    graph = MolGraph(mol, fg_mode="none")

    graph_tensors, _scopes = get_batch_graphs([graph], fg_mode="none")
    undirected_b2a = graph_tensors[-1]
    expected = [[0, 0]]
    expected.extend(
        sorted([bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1])
        for bond in sorted(mol.GetBonds(), key=lambda bond: bond.GetIdx())
    )

    assert undirected_b2a.tolist() == expected


def test_prepare_edit_labels_still_align_with_rdkit_bond_idx():
    mol = _manual_mol_with_non_pairscan_bond_indices()
    graph = MolGraph(mol, fg_mode="none")
    bond_vocab = Vocab([("Delete Bond", None)])
    atom_vocab = Vocab([("Change Atom", (0, 0, 0, 0))])
    first_rdkit_bond = mol.GetBondWithIdx(0)
    edit_atoms = sorted(
        [
            first_rdkit_bond.GetBeginAtom().GetAtomMapNum(),
            first_rdkit_bond.GetEndAtom().GetAtomMapNum(),
        ]
    )

    labels = prepare_edit_labels([graph], [("Delete Bond", None)], [edit_atoms], bond_vocab, atom_vocab)

    bond_label = labels[0][: graph.num_bonds * bond_vocab.size()].reshape(graph.num_bonds, bond_vocab.size())
    assert torch.equal(bond_label[:, 0], torch.tensor([1.0, 0.0, 0.0], dtype=bond_label.dtype))
