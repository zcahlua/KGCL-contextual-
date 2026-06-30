import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from kgcl_retro.chemistry.graphs import MolGraph
from kgcl_retro.data.collate import get_batch_graphs, prepare_edit_labels
from kgcl_retro.chemistry.graphs import Vocab


def test_profile_hooks_do_not_print_by_default(capsys):
    graph = MolGraph(Chem.MolFromSmiles("CCO"), fg_mode="contextual", fg_context_radius=1)
    get_batch_graphs([graph], fg_mode="contextual")

    captured = capsys.readouterr()

    assert "KGCL_PROFILE" not in captured.out


def test_profile_hooks_print_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("KGCL_PROFILE_GRAPH_BUILD", "1")
    monkeypatch.setenv("KGCL_PROFILE_CONTEXTUAL_FG", "1")
    mol = Chem.MolFromSmiles("CCO")
    graph = MolGraph(mol, fg_mode="contextual", fg_context_radius=1)
    get_batch_graphs([graph], fg_mode="contextual")
    prepare_edit_labels(
        [graph],
        ["Terminate"],
        [[]],
        Vocab([("Delete Bond", None)]),
        Vocab([("Change Atom", (0, 0, 0, 0))]),
    )

    captured = capsys.readouterr()

    assert "MolGraph build time" in captured.out
    assert "FG matching time" in captured.out
    assert "get_batch_graphs time" in captured.out
    assert "prepare_edit_labels time" in captured.out
