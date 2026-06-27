from argparse import Namespace

import pytest

from kgcl_retro.cli.fg_options import resolve_eval_fg_config


def _args(**overrides):
    args = {
        "fg_mode": None,
        "fg_context_radius": None,
        "fg_hidden_size": None,
        "fg_layers": None,
        "fg_attn_dim": None,
        "fg_max_dist": None,
        "fg_use_kg_fusion": None,
        "fg_use_membership_bias": None,
        "fg_use_distance_bias": None,
        "fg_null_token": None,
        "fg_freeze_kg_projection": None,
        "fg_freeze_kg_embeddings": None,
        "allow_architecture_override": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_eval_uses_checkpoint_fg_mode_when_cli_omits_it():
    resolved = resolve_eval_fg_config({"fg_mode": "contextual"}, _args())

    assert resolved["fg_mode"] == "contextual"


def test_eval_allows_matching_cli_fg_mode():
    resolved = resolve_eval_fg_config({"fg_mode": "contextual"}, _args(fg_mode="contextual_fg"))

    assert resolved["fg_mode"] == "contextual"


def test_eval_rejects_mismatched_cli_architecture_by_default():
    with pytest.raises(ValueError, match="fg_mode"):
        resolve_eval_fg_config({"fg_mode": "legacy"}, _args(fg_mode="contextual"))


def test_eval_treats_old_checkpoint_without_fg_mode_as_legacy():
    with pytest.raises(ValueError, match="fg_mode"):
        resolve_eval_fg_config({"mpn_size": 16, "dropout_mpn": 0.0}, _args(fg_mode="contextual"))


def test_eval_override_allows_mismatch_with_warning(capsys):
    resolved = resolve_eval_fg_config(
        {"fg_mode": "legacy"},
        _args(fg_mode="contextual", allow_architecture_override=True),
    )
    captured = capsys.readouterr()

    assert resolved["fg_mode"] == "contextual"
    assert "architecture override" in captured.out


def test_eval_freeze_alias_is_normalized():
    resolved = resolve_eval_fg_config(
        {"fg_mode": "contextual", "fg_freeze_kg_projection": True},
        _args(fg_freeze_kg_embeddings=True),
    )

    assert resolved["fg_freeze_kg_projection"] is True
