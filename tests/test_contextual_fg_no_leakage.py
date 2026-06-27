import inspect

from kgcl_retro.chemistry import fg_instances


def test_contextual_matching_api_does_not_accept_training_labels_or_reactants():
    signature = inspect.signature(fg_instances.match_fg_instances)
    forbidden = {"edit", "edit_atoms", "edit_labels", "reactants", "reac_mol", "labels"}

    assert forbidden.isdisjoint(signature.parameters)


def test_contextual_model_constructors_do_not_accept_training_labels_or_reactants():
    from kgcl_retro.models.contextual_fg import AtomFGAttention, ContextualFGGraphEncoder, KGContextFusion

    forbidden = {"edit", "edit_atoms", "edit_labels", "reactants", "reac_mol", "labels"}
    for cls in (ContextualFGGraphEncoder, KGContextFusion, AtomFGAttention):
        signature = inspect.signature(cls.__init__)
        assert forbidden.isdisjoint(signature.parameters)
