from kgcl_retro.models import BeamSearch, KGCL  # Explanation: imports packaged model exports for smoke testing.


def test_model_classes_import():  # Explanation: verifies the model package exposes the main model and decoder.
    assert KGCL.__name__ == "KGCL"  # Explanation: checks the main neural model export.
    assert BeamSearch.__name__ == "BeamSearch"  # Explanation: checks the beam-search decoder export.


def test_contextual_fg_legacy_import_paths():
    from models import AtomFGAttention, ContextualFGGraphEncoder, KGContextFusion
    from models.contextual_fg import ContextualFGGraphEncoder as LegacyContextualFGGraphEncoder
    from utils.fg_instances import FunctionalGroupInstance, match_fg_instances

    assert ContextualFGGraphEncoder.__name__ == "ContextualFGGraphEncoder"
    assert LegacyContextualFGGraphEncoder is ContextualFGGraphEncoder
    assert KGContextFusion.__name__ == "KGContextFusion"
    assert AtomFGAttention.__name__ == "AtomFGAttention"
    assert FunctionalGroupInstance.__name__ == "FunctionalGroupInstance"
    assert callable(match_fg_instances)
