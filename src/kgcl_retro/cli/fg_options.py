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
    "fg_freeze_kg_embeddings",
    "fg_debug",
)


def add_fg_arguments(parser: ArgumentParser, default_mode: str | None = "legacy") -> None:
    parser.add_argument("--fg_mode", choices=["legacy", "contextual", "contextual_fg", "none"], default=default_mode)
    parser.add_argument("--fg_context_radius", type=int, default=1)
    parser.add_argument("--fg_hidden_size", type=int, default=None)
    parser.add_argument("--fg_layers", type=int, default=2)
    parser.add_argument("--fg_dropout", type=float, default=None)
    parser.add_argument("--fg_attn_dim", type=int, default=None)
    parser.add_argument("--fg_max_dist", type=int, default=8)
    parser.add_argument("--fg_max_matches_per_pattern", type=int, default=None)
    parser.add_argument("--fg_use_kg_fusion", dest="fg_use_kg_fusion", action="store_true", default=True)
    parser.add_argument("--no_fg_use_kg_fusion", dest="fg_use_kg_fusion", action="store_false")
    parser.add_argument("--fg_use_membership_bias", dest="fg_use_membership_bias", action="store_true", default=True)
    parser.add_argument("--no_fg_use_membership_bias", dest="fg_use_membership_bias", action="store_false")
    parser.add_argument("--fg_use_distance_bias", dest="fg_use_distance_bias", action="store_true", default=True)
    parser.add_argument("--no_fg_use_distance_bias", dest="fg_use_distance_bias", action="store_false")
    parser.add_argument("--fg_null_token", dest="fg_null_token", action="store_true", default=True)
    parser.add_argument("--no_fg_null_token", dest="fg_null_token", action="store_false")
    parser.add_argument("--fg_freeze_kg_embeddings", default=False, action="store_true")
    parser.add_argument("--fg_debug", default=False, action="store_true")


def fg_config_from_args(args: dict[str, Any]) -> dict[str, Any]:
    return {key: args.get(key) for key in FG_CONFIG_KEYS if key in args}


def processed_fg_subdir(args: Any) -> str:
    fg_mode = getattr(args, "fg_mode", "legacy") if not isinstance(args, dict) else args.get("fg_mode", "legacy")
    radius = getattr(args, "fg_context_radius", 1) if not isinstance(args, dict) else args.get("fg_context_radius", 1)
    if fg_mode == "contextual_fg":
        fg_mode = "contextual"
    if fg_mode == "legacy":
        return ""
    if fg_mode == "contextual":
        return f"contextual_fg_r{radius}"
    return "fg_none"


def metadata_for_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        args = vars(args)
    metadata = fg_config_from_args(args)
    if metadata.get("fg_mode") == "contextual_fg":
        metadata["fg_mode"] = "contextual"
    metadata["use_rxn_class"] = bool(args.get("use_rxn_class", False))
    return metadata


def write_processed_metadata(savedir: str, args: Any) -> None:
    metadata = metadata_for_args(args)
    with open(os.path.join(savedir, "fg_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


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
    expected = metadata_for_args(args)
    for key in ("fg_mode", "fg_context_radius", "fg_max_dist", "fg_max_matches_per_pattern", "use_rxn_class"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(
                f"Prepared data mismatch for {key}: expected {expected.get(key)!r}, "
                f"found {metadata.get(key)!r}. Regenerate prepared data for this FG configuration."
            )
