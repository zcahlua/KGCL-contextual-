# Contextual Functional-Group KGCL

Contextual FG mode replaces KGCL's type-level functional-group memory with matched-instance graph encoding.

In legacy mode, KGCL does:

```text
SMARTS type exists in molecule -> KG embedding -> atom attention before D-MPNN
```

In contextual mode, KGCL does:

```text
SMARTS occurrence with atom indices
-> matched core C_g
-> k-hop induced local graph G_t[N_kappa(C_g)]
-> ContextualFGGraphEncoder
-> KG/context fusion
-> membership and distance biased atom-to-FG attention
-> D-MPNN and existing edit heads
```

Each functional-group instance stores:

```text
name
SMARTS pattern
pattern index
matched atom indices
core atom indices
KG embedding
product-computable chemistry descriptors
```

Functional-group overlaps are allowed. The matcher deduplicates automorphic duplicates only when the same functional-group name matches the same sorted atom set.

## Modes

- `--fg_mode legacy`: default, keeps original KGCL tensor behavior and checkpoint compatibility.
- `--fg_mode contextual`: uses matched-instance contextual FG graph encoding.
- `--fg_mode none`: skips functional-group fusion.

`--fg_mode contextual_fg` is accepted as an alias for `contextual`.

## Commands

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual --fg_context_radius 1
kgcl-train --dataset uspto_50k --fg_mode contextual --fg_context_radius 1
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual --fg_context_radius 1
```

Prepared contextual tensors are not interchangeable with legacy tensors. Contextual and none-mode batches are written into separate subdirectories and include `fg_metadata.json` so training can assert that the model mode matches the prepared data.
The metadata includes the requested FG options, reaction-class setting, atom and
bond feature dimensions, and a KG asset fingerprint. A mismatch fails with a
clear instruction to re-run `kgcl-prepare-data` with matching FG options.

Evaluation uses the FG architecture saved in the checkpoint unless an eval flag is supplied. A matching flag is accepted, while a mismatch raises by default:

```bash
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual
```

Use `--allow_architecture_override` only for deliberate checkpoint surgery or debugging, because contextual and legacy modes have different parameter sets. `atom_message=True` is rejected early; the implemented encoder path is directed-bond-message D-MPNN. `--fg_freeze_kg_projection` freezes only the projection from KG embeddings in `KGContextFusion`; the old `--fg_freeze_kg_embeddings` flag is a deprecated alias.

## Scope

This implementation does not implement sparse 2-FWL, PairWL, candidate nonbonded pair scoring, or any expansion of the bond-edit action space. Bond scores are still produced from existing bond endpoint atom embeddings in the same order expected by the current edit labels and beam search.

Contextual FG inputs use only the current product or intermediate molecule, SMARTS vocabulary, matched atom indices, product-computable descriptors, KG embeddings, and optional reaction class. They do not use gold edit labels, edit atoms, reactants, edit order labels, contrastive labels, or functional-group participation labels.

## Profiling and Caching

Set `KGCL_PROFILE_GRAPH_BUILD=1` to print MolGraph build time,
`get_batch_graphs` time, and `prepare_edit_labels` time. Set
`KGCL_PROFILE_CONTEXTUAL_FG=1` to print FG matching time, context BFS time, FG
instance counts, and average context size. Nothing is printed unless these
variables are set.

`KGCL_CONTEXTUAL_FG_CACHE=1` enables an optional contextual FG metadata cache in
`kgcl-prepare-data`. The cache key includes an atom-map-sensitive molecule key,
`use_rxn_class`, FG mode, context radius, max distance, match cap, and a KG asset
fingerprint. It caches matched instances, context atoms, distances, context
edges, KG embeddings, and product-computable descriptors. It does not cache
batch-offset tensors.

For a small local timing check that does not require USPTO data:

```bash
python -m kgcl_retro.benchmarks.contextual_fg_prepare --fg_mode contextual --use_cache
```

Dense edit-label allocation remains the default. A future experimental
`label_format=index` path should keep dense labels as the default until it is
measured against training compatibility.
