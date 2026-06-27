import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from kgcl_retro.chemistry.graphs import Vocab
from kgcl_retro.models.contextual_fg import KGContextFusion
from kgcl_retro.models.kgcl import KGCL, add_fg_config_defaults


def _contextual_config(**overrides):
    config = {
        "n_atom_feat": 85,
        "n_bond_feat": 97,
        "mpn_size": 16,
        "mlp_size": 32,
        "depth": 2,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": False,
        "use_attn": False,
        "n_heads": 2,
        "fg_mode": "contextual",
    }
    config.update(overrides)
    return config


def test_freeze_kg_projection_only_freezes_kg_projection():
    fusion = KGContextFusion(
        kg_dim=85,
        ctx_dim=16,
        chem_descriptor_dim=9,
        out_dim=16,
        freeze_kg_projection=True,
    )

    assert all(not param.requires_grad for param in fusion.kg_proj.parameters())
    assert all(param.requires_grad for param in fusion.ctx_proj.parameters())
    assert all(param.requires_grad for param in fusion.gate.parameters())


def test_deprecated_freeze_kg_embeddings_alias_maps_to_projection_freeze():
    config = add_fg_config_defaults(
        {
            "mpn_size": 16,
            "dropout_mpn": 0.0,
            "atom_message": False,
            "fg_freeze_kg_embeddings": True,
        }
    )

    assert config["fg_freeze_kg_projection"] is True


def test_conflicting_freeze_aliases_raise():
    with pytest.raises(ValueError, match="fg_freeze_kg_projection"):
        add_fg_config_defaults(
            {
                "mpn_size": 16,
                "dropout_mpn": 0.0,
                "atom_message": False,
                "fg_freeze_kg_projection": True,
                "fg_freeze_kg_embeddings": False,
            }
        )


def test_contextual_model_uses_projection_freeze_flag():
    model = KGCL(
        _contextual_config(fg_freeze_kg_projection=True),
        Vocab([("Change Atom", (0, 0, 0, 0))]),
        Vocab([("Delete Bond", None)]),
    )

    assert all(not param.requires_grad for param in model.kg_context_fusion.kg_proj.parameters())
