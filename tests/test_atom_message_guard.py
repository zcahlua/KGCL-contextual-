import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from kgcl_retro.chemistry.graphs import Vocab
from kgcl_retro.models.encoder import MPNEncoder
from kgcl_retro.models.kgcl import KGCL


def _config(atom_message):
    return {
        "n_atom_feat": 85,
        "n_bond_feat": 97,
        "mpn_size": 16,
        "mlp_size": 32,
        "depth": 2,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": atom_message,
        "use_attn": False,
        "n_heads": 2,
    }


def test_kgcl_rejects_atom_message_true():
    with pytest.raises(ValueError, match="atom_message=True is not implemented"):
        KGCL(_config(atom_message=True), Vocab([("Change Atom", (0, 0, 0, 0))]), Vocab([("Delete Bond", None)]))


def test_mpn_encoder_rejects_atom_message_true():
    with pytest.raises(ValueError, match="atom_message=True is not implemented"):
        MPNEncoder(atom_fdim=85, bond_fdim=97, hidden_size=16, depth=2, atom_message=True)


def test_atom_message_false_still_constructs():
    encoder = MPNEncoder(atom_fdim=85, bond_fdim=97, hidden_size=16, depth=2, atom_message=False)

    assert encoder.atom_message is False
