import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import kgcl_retro.chemistry.graphs as graph_module
from kgcl_retro.chemistry.fg_instances import FunctionalGroupInstance
from kgcl_retro.chemistry.graphs import MolGraph


def _mapped_ethanol(first_map=1):
    mol = Chem.MolFromSmiles("CCO")
    for offset, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(first_map + offset)
    return mol


def _fake_instance(mol):
    return FunctionalGroupInstance(
        name="fake",
        smarts="[C]",
        pattern_index=0,
        match_atoms=(0,),
        core_atoms=(0,),
        kg_embedding=[0.1, 0.2, 0.3],
        chem_descriptors=[0.0] * 9,
    )


def test_fg_metadata_cache_hit(monkeypatch):
    calls = {"count": 0}

    def fake_match(mol, use_rxn_class, max_matches_per_pattern=None):
        calls["count"] += 1
        return [_fake_instance(mol)]

    monkeypatch.setattr(graph_module, "match_fg_instances", fake_match)
    cache = {}
    mol = _mapped_ethanol()

    first = MolGraph(mol, fg_mode="contextual", fg_context_radius=1, fg_metadata_cache=cache)
    second = MolGraph(Chem.Mol(mol), fg_mode="contextual", fg_context_radius=1, fg_metadata_cache=cache)

    assert calls["count"] == 1
    assert first.fg_context_atom_indices == second.fg_context_atom_indices
    assert first.fg_instances is not second.fg_instances


def test_fg_metadata_cache_miss_when_radius_changes(monkeypatch):
    calls = {"count": 0}

    def fake_match(mol, use_rxn_class, max_matches_per_pattern=None):
        calls["count"] += 1
        return [_fake_instance(mol)]

    monkeypatch.setattr(graph_module, "match_fg_instances", fake_match)
    cache = {}
    mol = _mapped_ethanol()

    MolGraph(mol, fg_mode="contextual", fg_context_radius=0, fg_metadata_cache=cache)
    MolGraph(Chem.Mol(mol), fg_mode="contextual", fg_context_radius=1, fg_metadata_cache=cache)

    assert calls["count"] == 2


def test_fg_metadata_cache_preserves_atom_map_alignment(monkeypatch):
    calls = {"count": 0}

    def fake_match(mol, use_rxn_class, max_matches_per_pattern=None):
        calls["count"] += 1
        return [_fake_instance(mol)]

    monkeypatch.setattr(graph_module, "match_fg_instances", fake_match)
    cache = {}

    MolGraph(_mapped_ethanol(first_map=1), fg_mode="contextual", fg_context_radius=1, fg_metadata_cache=cache)
    MolGraph(_mapped_ethanol(first_map=101), fg_mode="contextual", fg_context_radius=1, fg_metadata_cache=cache)

    assert calls["count"] == 2


def test_cache_disabled_matches_uncached_output(monkeypatch):
    def fake_match(mol, use_rxn_class, max_matches_per_pattern=None):
        return [_fake_instance(mol)]

    monkeypatch.setattr(graph_module, "match_fg_instances", fake_match)
    mol = _mapped_ethanol()

    uncached = MolGraph(mol, fg_mode="contextual", fg_context_radius=1)
    cached = MolGraph(Chem.Mol(mol), fg_mode="contextual", fg_context_radius=1, fg_metadata_cache={})

    assert cached.fg_context_atom_indices == uncached.fg_context_atom_indices
    assert cached.fg_dist_to_core == uncached.fg_dist_to_core
    assert cached.atom_to_fg_membership == uncached.atom_to_fg_membership
    assert cached.atom_to_fg_dist == uncached.atom_to_fg_dist
