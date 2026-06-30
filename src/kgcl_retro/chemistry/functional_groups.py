from __future__ import annotations  # Explanation: defers annotation evaluation so optional chemistry types do not need runtime imports.

from dataclasses import dataclass  # Explanation: imports dataclass for the immutable resource container.
from functools import lru_cache  # Explanation: imports lru_cache so asset files are loaded once per embedding set.
import hashlib
from importlib import resources  # Explanation: imports package-resource helpers for installed asset files.
import pickle  # Explanation: imports pickle to load the saved functional-group embedding dictionary.
from typing import Any  # Explanation: imports Any for RDKit and embedding objects whose concrete types come from external packages.


@dataclass(frozen=True)  # Explanation: makes the loaded functional-group resources immutable after construction.
class FunctionalGroupResources:  # Explanation: groups SMARTS patterns, names, and KG embeddings for graph construction.
    names: list[str]  # Explanation: stores functional-group names in the same order as the SMARTS patterns.
    smarts_strings: list[str]  # Explanation: stores the original SMARTS text for matched-instance metadata.
    smarts: list[Any]  # Explanation: stores compiled RDKit SMARTS molecules used for substructure matching.
    smarts_to_name: dict[Any, str]  # Explanation: maps each compiled SMARTS object back to its functional-group name.
    embeddings: dict[str, Any]  # Explanation: maps each functional-group name to its knowledge-graph embedding vector.


def _asset_root(embedding_set: str):  # Explanation: resolves the package asset directory for one embedding set.
    return resources.files("kgcl_retro").joinpath("assets", embedding_set)  # Explanation: returns a Traversable path inside the installed package.


@lru_cache(maxsize=2)  # Explanation: caches both KGembedding variants used by the paper's ablations.
def load_functional_group_resources(embedding_set: str) -> FunctionalGroupResources:  # Explanation: loads names, SMARTS patterns, and embeddings for KGCL graph features.
    from rdkit import Chem  # Explanation: imports RDKit lazily so package import still works before chemistry dependencies are installed.

    root = _asset_root(embedding_set)  # Explanation: locates the requested packaged embedding directory.
    funcgroup_text = root.joinpath("funcgroup.txt").read_text()  # Explanation: reads the functional-group SMARTS definition file.
    rows = [line.split() for line in funcgroup_text.strip().splitlines()]  # Explanation: tokenizes each functional-group row into name and SMARTS pattern.
    names = [row[0] for row in rows]  # Explanation: extracts the functional-group names used as embedding keys.
    smarts_strings = [row[1] for row in rows]  # Explanation: keeps the textual SMARTS pattern for contextual FG diagnostics.
    smarts = [Chem.MolFromSmarts(pattern) for pattern in smarts_strings]  # Explanation: compiles SMARTS strings into RDKit substructure query molecules.
    with root.joinpath("fg2emb.pkl").open("rb") as handle:  # Explanation: opens the packaged embedding pickle in binary mode.
        embeddings = pickle.load(handle)  # Explanation: deserializes the functional-group embedding dictionary.
    return FunctionalGroupResources(  # Explanation: returns one structured resource object to graph-building code.
        names=names,  # Explanation: preserves the functional-group names for diagnostics and tests.
        smarts_strings=smarts_strings,  # Explanation: passes raw SMARTS text to contextual instance metadata.
        smarts=smarts,  # Explanation: passes compiled SMARTS patterns to substructure matching.
        smarts_to_name=dict(zip(smarts, names)),  # Explanation: builds the reverse lookup used after a SMARTS match.
        embeddings=embeddings,  # Explanation: passes the KG embedding vectors to graph feature fusion.
    )  # Explanation: closes construction of the functional-group resource container.


@lru_cache(maxsize=2)
def get_functional_group_asset_metadata(embedding_set: str = "KGembedding") -> tuple[int, int, list[str]]:
    root = _asset_root(embedding_set)
    funcgroup_text = root.joinpath("funcgroup.txt").read_text()
    names = [line.split()[0] for line in funcgroup_text.strip().splitlines()]
    with root.joinpath("fg2emb.pkl").open("rb") as handle:
        embeddings = pickle.load(handle)
    first_embedding = next(iter(embeddings.values()))
    return len(names), len(first_embedding), names


@lru_cache(maxsize=2)
def get_functional_group_asset_fingerprint(embedding_set: str = "KGembedding") -> dict[str, Any]:
    root = _asset_root(embedding_set)
    funcgroup_bytes = root.joinpath("funcgroup.txt").read_bytes()
    fg2emb_bytes = root.joinpath("fg2emb.pkl").read_bytes()
    names = [line.split()[0] for line in funcgroup_bytes.decode("utf-8").strip().splitlines()]
    return {
        "embedding_set": embedding_set,
        "num_fg_types": len(names),
        "funcgroup_sha256": hashlib.sha256(funcgroup_bytes).hexdigest(),
        "fg2emb_sha256": hashlib.sha256(fg2emb_bytes).hexdigest(),
    }


def match_fg_instances(mol: Any, use_rxn_class: bool, max_matches_per_pattern: int | None = None):
    from kgcl_retro.chemistry.fg_instances import match_fg_instances as _match_fg_instances

    return _match_fg_instances(mol, use_rxn_class, max_matches_per_pattern)
