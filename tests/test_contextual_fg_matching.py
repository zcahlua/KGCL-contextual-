import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from kgcl_retro.chemistry.fg_instances import match_fg_instances


def test_match_fg_instances_returns_atom_index_occurrences():
    mol = Chem.MolFromSmiles("CCOCC")

    instances = match_fg_instances(mol, use_rxn_class=False)

    assert instances
    assert all(instance.match_atoms for instance in instances)
    assert all(isinstance(idx, int) for instance in instances for idx in instance.match_atoms)
    assert all(instance.core_atoms == instance.match_atoms for instance in instances)


def test_multiple_occurrences_of_same_type_are_separate_instances():
    mol = Chem.MolFromSmiles("ClCCCl")

    instances = match_fg_instances(mol, use_rxn_class=False)
    chloro_instances = [instance for instance in instances if instance.name == "chloro"]

    assert len(chloro_instances) == 2
    assert {instance.match_atoms for instance in chloro_instances} == {(0,), (3,)}


def test_automorphic_duplicate_matches_are_removed_by_atom_set():
    mol = Chem.MolFromSmiles("CC")

    instances = match_fg_instances(mol, use_rxn_class=False)
    keys = [(instance.name, tuple(sorted(instance.match_atoms))) for instance in instances]

    assert len(keys) == len(set(keys))


def test_overlapping_functional_group_instances_are_allowed():
    mol = Chem.MolFromSmiles("CC(=O)O")

    instances = match_fg_instances(mol, use_rxn_class=False)
    carbonyl = [instance for instance in instances if instance.name == "Carbonyl"]
    carboxyl = [instance for instance in instances if instance.name == "Carboxyl"]

    assert carbonyl
    assert carboxyl
    assert set(carbonyl[0].core_atoms) & set(carboxyl[0].core_atoms)
