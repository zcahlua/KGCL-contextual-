from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from typing import Any


FG_CONFIG_KEYS = (
    "fg_mode",
    "fg_context_radius",
    "fg_hidden_size",
    "fg_layers",
    "fg_dropout",
    "fg_attn_dim",
    "fg_max_dist",
    "fg_max_matches_per_pattern",
    "fg_use_kg_fusion",
    "fg_use_membership_bias",
    "fg_use_distance_bias",
    "fg_null_token",
    "fg_freeze_kg_projection",
    "fg_freeze_kg_embeddings",
    "fg_debug",
)

ARCHITECTURE_FG_KEYS = tuple(key for key in FG_CONFIG_KEYS if key not in {"fg_debug", "fg_freeze_kg_embeddings"})
ARCHITECTURE_KEYS = (
    "fg_mode",
    "fg_context_radius",
    "fg_hidden_size",
    "fg_layers",
    "fg_dropout",
    "fg_attn_dim",
    "fg_max_dist",
    "fg_max_matches_per_pattern",
    "fg_use_kg_fusion",
    "fg_use_membership_bias",
    "fg_use_distance_bias",
    "fg_null_token",
    "fg_freeze_kg_projection",
    "use_rxn_class",
    "atom_message",
    "n_atom_feat",
    "n_bond_feat",
    "mpn_size",
    "depth",
)
_PREPARED_DATA_KEYS = (
    "fg_mode",
    "fg_context_radius",
    "fg_max_dist",
    "fg_max_matches_per_pattern",
    "use_rxn_class",
)
_VALID_FG_MODES = {"legacy", "contextual", "none"}


def _as_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    return vars(args)


def _canonical_fg_mode(mode: Any) -> Any:
    if mode is None:
        return None
    if mode == "contextual_fg":
        return "contextual"
    if mode not in _VALID_FG_MODES:
        raise ValueError(f"Unsupported fg_mode: {mode}")
    return mode


def normalize_fg_config_values(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "fg_mode" in normalized:
        normalized["fg_mode"] = _canonical_fg_mode(normalized["fg_mode"])

    projection_present = normalized.get("fg_freeze_kg_projection") is not None
    embeddings_present = normalized.get("fg_freeze_kg_embeddings") is not None
    if projection_present and embeddings_present:
        if bool(normalized["fg_freeze_kg_projection"]) != bool(normalized["fg_freeze_kg_embeddings"]):
            raise ValueError(
                "Conflicting FG freeze flags: fg_freeze_kg_projection and deprecated "
                "fg_freeze_kg_embeddings differ."
            )
    if projection_present:
        freeze = bool(normalized["fg_freeze_kg_projection"])
        normalized["fg_freeze_kg_projection"] = freeze
        normalized["fg_freeze_kg_embeddings"] = freeze
    elif embeddings_present:
        freeze = bool(normalized["fg_freeze_kg_embeddings"])
        normalized["fg_freeze_kg_projection"] = freeze
        normalized["fg_freeze_kg_embeddings"] = freeze

    return normalized


def _fill_checkpoint_comparison_defaults(config: dict[str, Any]) -> dict[str, Any]:
    filled = dict(config)
    filled.setdefault("fg_mode", "legacy")
    filled.setdefault("fg_context_radius", 1)
    filled.setdefault("fg_layers", 2)
    filled.setdefault("fg_max_dist", 8)
    filled.setdefault("fg_max_matches_per_pattern", None)
    filled.setdefault("fg_use_kg_fusion", True)
    filled.setdefault("fg_use_membership_bias", True)
    filled.setdefault("fg_use_distance_bias", True)
    filled.setdefault("fg_null_token", True)
    filled.setdefault("fg_freeze_kg_projection", False)
    filled.setdefault("fg_freeze_kg_embeddings", filled["fg_freeze_kg_projection"])
    filled.setdefault("use_rxn_class", False)
    filled.setdefault("atom_message", False)
    if filled.get("fg_hidden_size") is None and filled.get("mpn_size") is not None:
        filled["fg_hidden_size"] = filled["mpn_size"]
    if filled.get("fg_attn_dim") is None and filled.get("mpn_size") is not None:
        filled["fg_attn_dim"] = filled["mpn_size"]
    if filled.get("fg_dropout") is None and filled.get("dropout_mpn") is not None:
        filled["fg_dropout"] = filled["dropout_mpn"]
    return normalize_fg_config_values(filled)


def add_fg_arguments(parser: ArgumentParser, default_mode: str | None = "legacy") -> None:
    eval_defaults = default_mode is None
    int_default = None if eval_defaults else 1
    layers_default = None if eval_defaults else 2
    max_dist_default = None if eval_defaults else 8
    true_default = None if eval_defaults else True
    debug_default = None if eval_defaults else False

    parser.add_argument("--fg_mode", choices=["legacy", "contextual", "contextual_fg", "none"], default=default_mode)
    parser.add_argument("--fg_context_radius", type=int, default=int_default)
    parser.add_argument("--fg_hidden_size", type=int, default=None)
    parser.add_argument("--fg_layers", type=int, default=layers_default)
    parser.add_argument("--fg_dropout", type=float, default=None)
    parser.add_argument("--fg_attn_dim", type=int, default=None)
    parser.add_argument("--fg_max_dist", type=int, default=max_dist_default)
    parser.add_argument("--fg_max_matches_per_pattern", type=int, default=None)
    parser.add_argument("--fg_use_kg_fusion", dest="fg_use_kg_fusion", action="store_true", default=true_default)
    parser.add_argument("--no_fg_use_kg_fusion", dest="fg_use_kg_fusion", action="store_false")
    parser.add_argument("--fg_use_membership_bias", dest="fg_use_membership_bias", action="store_true", default=true_default)
    parser.add_argument("--no_fg_use_membership_bias", dest="fg_use_membership_bias", action="store_false")
    parser.add_argument("--fg_use_distance_bias", dest="fg_use_distance_bias", action="store_true", default=true_default)
    parser.add_argument("--no_fg_use_distance_bias", dest="fg_use_distance_bias", action="store_false")
    parser.add_argument("--fg_null_token", dest="fg_null_token", action="store_true", default=true_default)
    parser.add_argument("--no_fg_null_token", dest="fg_null_token", action="store_false")
    parser.add_argument("--fg_freeze_kg_projection", default=None, action="store_true")
    parser.add_argument("--fg_freeze_kg_embeddings", default=None, action="store_true")
    parser.add_argument("--fg_debug", default=debug_default, action="store_true")


def fg_config_from_args(args: dict[str, Any]) -> dict[str, Any]:
    return normalize_fg_config_values({key: args.get(key) for key in FG_CONFIG_KEYS if key in args})


def processed_fg_subdir(args: Any) -> str:
    args_dict = _as_dict(args)
    fg_mode = _canonical_fg_mode(args_dict.get("fg_mode", "legacy"))
    radius = args_dict.get("fg_context_radius", 1)
    if fg_mode == "legacy":
        return ""
    if fg_mode == "contextual":
        return f"contextual_fg_r{radius}"
    return "fg_none"


def metadata_for_args(args: Any) -> dict[str, Any]:
    args = _as_dict(args)
    metadata = fg_config_from_args(args)
    metadata["fg_mode"] = _canonical_fg_mode(metadata.get("fg_mode", "legacy"))
    metadata["use_rxn_class"] = bool(args.get("use_rxn_class", False))
    return metadata


def write_processed_metadata(savedir: str, args: Any) -> None:
    metadata = metadata_for_args(args)
    with open(os.path.join(savedir, "fg_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def assert_prepared_data_compatible(config: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    expected = normalize_fg_config_values(dict(config))
    expected["fg_mode"] = _canonical_fg_mode(expected.get("fg_mode", "legacy"))
    if expected["fg_mode"] == "legacy":
        return
    if metadata is None:
        raise ValueError(
            "Prepared data has no fg_metadata. Regenerate it with kgcl-prepare-data "
            "using the same --fg_mode settings."
        )
    found = normalize_fg_config_values(dict(metadata))
    found["fg_mode"] = _canonical_fg_mode(found.get("fg_mode", "legacy"))

    defaults = {
        "fg_context_radius": 1,
        "fg_max_dist": 8,
        "fg_max_matches_per_pattern": None,
        "use_rxn_class": False,
    }
    for key in _PREPARED_DATA_KEYS:
        expected_value = expected.get(key, defaults.get(key))
        found_value = found.get(key, defaults.get(key))
        if expected_value != found_value:
            raise ValueError(
                f"Prepared data mismatch for {key}: expected {expected_value!r}, "
                f"found {found_value!r}. Regenerate prepared data for this FG configuration."
            )


def assert_processed_metadata(savedir: str, args: dict[str, Any]) -> None:
    variant = processed_fg_subdir(args)
    metadata_path = os.path.join(savedir, "fg_metadata.json")
    if not variant:
        return
    if not os.path.exists(metadata_path):
        raise ValueError(
            f"Prepared data in {savedir} has no fg_metadata.json. "
            "Regenerate it with kgcl-prepare-data using the same --fg_mode settings."
        )
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    assert_prepared_data_compatible(metadata_for_args(args), metadata)


def resolve_eval_fg_config(checkpoint_config: dict[str, Any], args: Any) -> dict[str, Any]:
    resolved = _fill_checkpoint_comparison_defaults(normalize_fg_config_values(dict(checkpoint_config)))
    args_dict = _as_dict(args)
    cli_config = {key: args_dict.get(key) for key in FG_CONFIG_KEYS if args_dict.get(key) is not None}
    cli_config = normalize_fg_config_values(cli_config)
    for key in ARCHITECTURE_KEYS:
        if key not in FG_CONFIG_KEYS and args_dict.get(key) is not None:
            cli_config[key] = args_dict[key]
    allow_override = bool(args_dict.get("allow_architecture_override", False))

    for key, value in cli_config.items():
        if key == "fg_freeze_kg_embeddings":
            continue
        if key not in ARCHITECTURE_KEYS:
            resolved[key] = value
            continue
        checkpoint_value = resolved.get(key)
        if checkpoint_value is None or checkpoint_value == value:
            resolved[key] = value if checkpoint_value is None else checkpoint_value
            continue
        if not allow_override:
            raise ValueError(
                f"Evaluation FG config mismatch for {key}: checkpoint has {checkpoint_value!r}, "
                f"CLI requested {value!r}. Omit the flag to use the checkpoint config, or pass "
                "--allow_architecture_override only when you intentionally load with a different architecture."
            )
        print(
            f"Warning: applying architecture override for {key}: checkpoint has "
            f"{checkpoint_value!r}, CLI requested {value!r}."
        )
        resolved[key] = value

    return normalize_fg_config_values(resolved)
