import pytest

from kgcl_retro.cli.fg_options import assert_prepared_data_compatible


def test_matching_contextual_metadata_passes():
    config = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": False}
    metadata = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": False}

    assert_prepared_data_compatible(config, metadata)


def test_contextual_config_without_metadata_fails():
    config = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": False}

    with pytest.raises(ValueError, match="fg_metadata"):
        assert_prepared_data_compatible(config, None)


def test_radius_mismatch_fails():
    config = {"fg_mode": "contextual", "fg_context_radius": 2, "fg_max_dist": 8, "use_rxn_class": False}
    metadata = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": False}

    with pytest.raises(ValueError, match="fg_context_radius"):
        assert_prepared_data_compatible(config, metadata)


def test_use_rxn_class_mismatch_fails():
    config = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": True}
    metadata = {"fg_mode": "contextual", "fg_context_radius": 1, "fg_max_dist": 8, "use_rxn_class": False}

    with pytest.raises(ValueError, match="use_rxn_class"):
        assert_prepared_data_compatible(config, metadata)
