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

## Scope

This implementation does not implement sparse 2-FWL, PairWL, candidate nonbonded pair scoring, or any expansion of the bond-edit action space. Bond scores are still produced from existing bond endpoint atom embeddings in the same order expected by the current edit labels and beam search.

Contextual FG inputs use only the current product or intermediate molecule, SMARTS vocabulary, matched atom indices, product-computable descriptors, KG embeddings, and optional reaction class. They do not use gold edit labels, edit atoms, reactants, edit order labels, contrastive labels, or functional-group participation labels.
