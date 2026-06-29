from __future__ import annotations

import argparse
import time


SMALL_MOLECULES = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN(CC)CC",
    "CCOC(=O)C",
    "CC(C)O",
    "CC(=O)NC",
    "COc1ccccc1",
    "CCCl",
    "CCBr",
    "CCS",
    "O=C(N)C",
    "CC(C)(C)O",
    "c1ccncc1",
    "CC(=O)OC",
)


def _time_graph_builds(smiles: tuple[str, ...], fg_mode: str, repeats: int, use_cache: bool) -> dict[str, float]:
    from rdkit import Chem

    from kgcl_retro.chemistry.graphs import MolGraph
    from kgcl_retro.data.collate import get_batch_graphs

    cache = {} if use_cache else None
    graphs = []
    start = time.perf_counter()
    for _ in range(repeats):
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            graphs.append(
                MolGraph(
                    mol,
                    fg_mode=fg_mode,
                    fg_context_radius=1,
                    fg_metadata_cache=cache,
                )
            )
    graph_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    get_batch_graphs(graphs, fg_mode=fg_mode)
    batch_ms = (time.perf_counter() - start) * 1000.0

    fg_counts = [len(getattr(graph, "fg_instances", [])) for graph in graphs]
    context_counts = [
        len(context_atoms)
        for graph in graphs
        for context_atoms in getattr(graph, "fg_context_atom_indices", [])
    ]
    return {
        "molecules": float(len(graphs)),
        "graph_build_ms": graph_ms,
        "batch_ms": batch_ms,
        "avg_fg_instances": sum(fg_counts) / max(len(fg_counts), 1),
        "avg_context_atoms": sum(context_counts) / max(len(context_counts), 1),
        "cache_entries": float(len(cache or {})),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual FG graph preparation on small molecules.")
    parser.add_argument("--fg_mode", choices=["legacy", "contextual", "none"], default="contextual")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args(argv)

    try:
        results = _time_graph_builds(SMALL_MOLECULES, args.fg_mode, args.repeats, args.use_cache)
    except ModuleNotFoundError as exc:
        if exc.name == "rdkit":
            raise SystemExit("RDKit is required to run this benchmark.") from exc
        raise

    print("contextual_fg_prepare benchmark")
    for key, value in results.items():
        if key.endswith("_ms") or key.startswith("avg_"):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {int(value)}")


if __name__ == "__main__":
    main()
