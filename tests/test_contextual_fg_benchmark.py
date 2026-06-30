def test_contextual_fg_prepare_benchmark_imports_without_rdkit_side_effects():
    from kgcl_retro.benchmarks import contextual_fg_prepare

    assert callable(contextual_fg_prepare.main)
