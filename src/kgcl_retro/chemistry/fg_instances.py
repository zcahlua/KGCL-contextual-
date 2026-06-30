from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kgcl_retro.chemistry.functional_groups import load_functional_group_resources


FG_CHEM_DESCRIPTOR_SIZE = 9


@dataclass
class FunctionalGroupInstance:
    name: str
    smarts: str
    pattern_index: int
    match_atoms: tuple[int, ...]
    core_atoms: tuple[int, ...]
    kg_embedding: list[float] | np.ndarray
    chem_descriptors: list[float]
    is_null: bool = False


def _rough_donor_acceptor_flags(mol: Any, atom_indices: tuple[int, ...]) -> tuple[float, float]:
    donor = 0.0
    acceptor = 0.0
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        if atomic_num in {7, 8, 16} and atom.GetTotalNumHs() > 0:
            donor = 1.0
        if atomic_num in {7, 8, 16} and formal_charge <= 0:
            acceptor = 1.0
    return donor, acceptor


def compute_fg_chem_descriptors(
    mol: Any,
    core_atoms: tuple[int, ...],
    context_atoms: tuple[int, ...] | None = None,
) -> list[float]:
    if not core_atoms:
        return [0.0] * FG_CHEM_DESCRIPTOR_SIZE

    context_atom_set = set(context_atoms or core_atoms)
    core_atom_set = set(core_atoms)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in core_atoms]
    aromatic_count = sum(float(atom.GetIsAromatic()) for atom in atoms)
    ring_count = sum(float(atom.IsInRing()) for atom in atoms)
    formal_charge_sum = sum(float(atom.GetFormalCharge()) for atom in atoms)
    hetero_atom_count = sum(float(atom.GetAtomicNum() not in {1, 6}) for atom in atoms)
    donor_flag, acceptor_flag = _rough_donor_acceptor_flags(mol, core_atoms)

    return [
        float(len(core_atoms)),
        float(len(context_atom_set - core_atom_set)),
        float(aromatic_count > 0),
        float(aromatic_count / max(len(core_atoms), 1)),
        float(ring_count > 0),
        formal_charge_sum,
        hetero_atom_count,
        donor_flag,
        acceptor_flag,
    ]


def match_fg_instances(
    mol: Any,
    use_rxn_class: bool,
    max_matches_per_pattern: int | None = None,
) -> list[FunctionalGroupInstance]:
    embedding_set = "KGembedding_2" if use_rxn_class else "KGembedding"
    resources = load_functional_group_resources(embedding_set)
    instances: list[FunctionalGroupInstance] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for pattern_index, query in enumerate(resources.smarts):
        if query is None:
            continue
        name = resources.names[pattern_index]
        smarts = resources.smarts_strings[pattern_index]
        matches = mol.GetSubstructMatches(query, uniquify=True)
        if max_matches_per_pattern is not None:
            matches = matches[:max_matches_per_pattern]

        for match in matches:
            match_atoms = tuple(int(atom_idx) for atom_idx in match)
            dedup_key = (name, tuple(sorted(match_atoms)))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            kg_embedding = resources.embeddings[name]
            if hasattr(kg_embedding, "tolist"):
                kg_embedding = kg_embedding.tolist()
            instances.append(
                FunctionalGroupInstance(
                    name=name,
                    smarts=smarts,
                    pattern_index=pattern_index,
                    match_atoms=match_atoms,
                    core_atoms=match_atoms,
                    kg_embedding=kg_embedding,
                    chem_descriptors=compute_fg_chem_descriptors(mol, match_atoms),
                )
            )

    return instances
