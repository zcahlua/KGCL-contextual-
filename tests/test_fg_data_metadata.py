import pytest

from kgcl_retro.cli.fg_options import assert_prepared_data_compatible, metadata_for_args


def _full_metadata(**overrides):
    metadata = {
        "fg_mode": "contextual",
        "fg_context_radius": 1,
        "fg_hidden_size": None,
        "fg_layers": 2,
        "fg_dropout": None,
        "fg_attn_dim": None,
        "fg_max_dist": 8,
        "fg_max_matches_per_pattern": None,
        "fg_use_kg_fusion": True,
        "fg_use_membership_bias": True,
        "fg_use_distance_bias": True,
        "fg_null_token": True,
        "fg_freeze_kg_projection": False,
        "fg_freeze_kg_embeddings": False,
        "use_rxn_class": False,
        "atom_fdim": 85,
        "bond_fdim": 12,
        "kg_asset_metadata": {
            "embedding_set": "KGembedding",
            "num_fg_types": 0,
            "funcgroup_sha256": "test",
            "fg2emb_sha256": "test",
        },
    }
    metadata.update(overrides)
    return metadata


def test_matching_contextual_metadata_passes():
    config = _full_metadata()
    metadata = _full_metadata()

    assert_prepared_data_compatible(config, metadata)


def test_contextual_config_without_metadata_fails():
    config = _full_metadata()

    with pytest.raises(ValueError, match="fg_metadata"):
        assert_prepared_data_compatible(config, None)


def test_radius_mismatch_fails():
    config = _full_metadata(fg_context_radius=2)
    metadata = _full_metadata(fg_context_radius=1)

    with pytest.raises(ValueError, match="Prepared data FG metadata does not match"):
        assert_prepared_data_compatible(config, metadata)


def test_use_rxn_class_mismatch_fails():
    config = _full_metadata(use_rxn_class=True)
    metadata = _full_metadata(use_rxn_class=False)

    with pytest.raises(ValueError, match="Prepared data FG metadata does not match"):
        assert_prepared_data_compatible(config, metadata)


def test_contextual_prepared_data_metadata_contains_fg_architecture_and_assets():
    metadata = metadata_for_args(
        {
            "fg_mode": "contextual_fg",
            "fg_context_radius": 2,
            "fg_hidden_size": 64,
            "fg_layers": 3,
            "fg_dropout": 0.1,
            "fg_attn_dim": 32,
            "fg_max_dist": 6,
            "fg_max_matches_per_pattern": 4,
            "fg_use_kg_fusion": False,
            "fg_use_membership_bias": True,
            "fg_use_distance_bias": False,
            "fg_null_token": True,
            "fg_freeze_kg_projection": True,
            "use_rxn_class": False,
        }
    )

    assert metadata["fg_mode"] == "contextual"
    assert metadata["fg_context_radius"] == 2
    assert metadata["fg_hidden_size"] == 64
    assert metadata["fg_use_kg_fusion"] is False
    assert metadata["fg_freeze_kg_projection"] is True
    assert metadata["fg_freeze_kg_embeddings"] is True
    assert "atom_fdim" in metadata
    assert "bond_fdim" in metadata
    assert metadata["kg_asset_metadata"]["embedding_set"] == "KGembedding"
    assert metadata["kg_asset_metadata"]["funcgroup_sha256"]
    assert metadata["kg_asset_metadata"]["fg2emb_sha256"]


def test_missing_contextual_metadata_key_fails_clearly():
    metadata = _full_metadata()
    metadata.pop("fg_use_distance_bias")

    with pytest.raises(ValueError, match="fg_use_distance_bias"):
        assert_prepared_data_compatible(_full_metadata(), metadata)
