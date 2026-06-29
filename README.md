# KGCL

KGCL is a retrosynthesis model based on molecular graph editing.

Paper title:

```text
KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction
Based on Molecular Graph Editing
```

This refactor keeps the original KGCL behavior available and adds an optional
contextual functional-group mode.

## Quick Start

Install the package from the repository root:

```bash
python -m pip install -e ".[dev]"
```

If RDKit is not available in your Python environment, install it first:

```bash
conda install -c conda-forge rdkit
python -m pip install -e ".[dev]"
```

Run the default legacy KGCL workflow on USPTO-50K:

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test

kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

The legacy top-level scripts still work, but the `kgcl-*` package commands are
preferred for new runs.

## Example: Contextual Functional Groups

Contextual mode must use data prepared with the same FG settings as training.
This example trains contextual FG KGCL with a 1-hop functional-group context:

```bash
# 1. Build edit labels and graph-edit sequences.
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test

# 2. Prepare contextual training tensors.
kgcl-prepare-data \
  --dataset uspto_50k \
  --mode train \
  --fg_mode contextual \
  --fg_context_radius 1

# 3. Train the contextual model.
kgcl-train \
  --dataset uspto_50k \
  --fg_mode contextual \
  --fg_context_radius 1

# 4. Evaluate with the same FG architecture.
kgcl-eval-50k \
  --dataset uspto_50k \
  --fg_mode contextual \
  --fg_context_radius 1
```

For reaction-class conditioning, add `--use_rxn_class` to the prepare, train,
and eval commands.

## Functional-Group Modes

Use `--fg_mode` to choose how KGCL uses functional groups:

| Mode | Meaning |
| --- | --- |
| `legacy` | Original KGCL behavior. SMARTS matches are reduced to FG types, KG embeddings are looked up, and non-trainable attention is applied in `MolGraph` before D-MPNN encoding. This is the default. |
| `contextual` | Each SMARTS occurrence is kept with atom indices, expanded to a k-hop local subgraph, encoded by a trainable contextual FG graph encoder, fused with the KG embedding, and injected through membership/distance-biased atom-to-FG attention. |
| `contextual_fg` | Alias for `contextual`. |
| `none` | Disables functional-group fusion for ablation. |

Common contextual options:

```bash
--fg_context_radius 1
--fg_hidden_size 256          # default: mpn_size
--fg_layers 2
--fg_dropout 0.15             # default: dropout_mpn
--fg_attn_dim 256             # default: mpn_size
--fg_max_dist 8
--fg_max_matches_per_pattern 32
--fg_use_kg_fusion / --no_fg_use_kg_fusion
--fg_use_membership_bias / --no_fg_use_membership_bias
--fg_use_distance_bias / --no_fg_use_distance_bias
--fg_null_token / --no_fg_null_token
--fg_freeze_kg_projection
```

Prepared batches for `contextual` and `none` modes are written into
mode-specific subdirectories, such as `contextual_fg_r1/`, and include
`fg_metadata.json`. Training checks this metadata so legacy and contextual
tensors are not mixed silently. Contextual metadata records the FG option set,
atom/bond feature dimensions, reaction-class setting, and a fingerprint of the
KG asset table. If those settings do not match, re-run `kgcl-prepare-data` with
the same FG options used for training.

Evaluation loads the FG architecture saved in the checkpoint by default.
Passing FG architecture flags during eval acts as a compatibility check.
Mismatches raise unless `--allow_architecture_override` is supplied
intentionally.

This implementation adds the Contextual Functional-Group KGCL stage only.
Sparse 2-FWL, PairWL, candidate nonbonded pair scoring, and bond-edit action
space expansion are not implemented here.

## Contextual FG Diagnostics

Graph preparation is quiet by default. For local profiling, set:

```bash
KGCL_PROFILE_GRAPH_BUILD=1
KGCL_PROFILE_CONTEXTUAL_FG=1
```

These print MolGraph build time, contextual FG matching/BFS timing, batch graph
collation timing, edit-label timing, FG instance counts, and average context
size. Contextual FG metadata caching can be enabled during data preparation
with:

```bash
KGCL_CONTEXTUAL_FG_CACHE=1 kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual
```

The cache key includes atom-map-sensitive molecule identity, reaction-class
setting, FG context settings, and the KG asset fingerprint. It caches only
graph-local FG metadata, not batch-offset tensors.

Run a lightweight local benchmark without USPTO data:

```bash
python -m kgcl_retro.benchmarks.contextual_fg_prepare --fg_mode contextual --use_cache
```

Dense one-hot edit labels remain the default for checkpoint compatibility. An
index-label target format can be added later as an experimental path once it is
benchmarked against the current dense training loop.

## Model Options

Core KGCL model options:

```bash
--mpn_size 256
--depth 10
--dropout_mpn 0.15
--mlp_size 512
--dropout_mlp 0.2
--use_attn
--n_heads 8
--use_rxn_class
```

`atom_message=True` is not implemented in this refactor. Use the default
directed-bond-message D-MPNN path.

## Data

The original datasets are from:

- USPTO-50K: <https://github.com/Hanjun-Dai/GLN> (`schneider50k`)
- USPTO-FULL: <https://github.com/Hanjun-Dai/GLN> (`uspto_multi`)

The raw and processed data can be downloaded from:
<https://drive.google.com/drive/folders/11YMNrm7St-GgVF278orHSXk-EKM3ltqH?usp=sharing>

Expected directory layout:

```text
KGCL/
  data/
    uspto_50k/
      canonicalized_test.csv
      canonicalized_train.csv
      canonicalized_valid.csv
      raw_test.csv
      raw_train.csv
      raw_valid.csv
    uspto_full/
      canonicalized_test.csv
      canonicalized_train.csv
      canonicalized_valid.csv
      raw_test.csv
      raw_train.csv
      raw_valid.csv
```

Raw files are named `raw_train.csv`, `raw_valid.csv`, and `raw_test.csv`.
Processed files are named `canonicalized_train.csv`,
`canonicalized_valid.csv`, and `canonicalized_test.csv`.

Some downloaded copies use `val` instead of `valid` in validation filenames.
The current CLIs use `--mode valid`, so rename or copy `raw_val.csv` to
`raw_valid.csv` before running the commands above.

## Package Commands

Canonicalize a split:

```bash
kgcl-canonicalize --dataset uspto_50k --mode train
kgcl-canonicalize --dataset uspto_50k --mode valid
kgcl-canonicalize --dataset uspto_50k --mode test
```

Preprocess a split:

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test
```

Prepare training tensors:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class
```

Train:

```bash
kgcl-train --dataset uspto_50k
kgcl-train --dataset uspto_50k --use_rxn_class
```

Evaluate:

```bash
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
kgcl-eval-full --dataset uspto_full
kgcl-eval-roundtrip --dataset uspto_50k
```

Trained checkpoints are saved under:

```text
experiments/<dataset>/without_rxn_class/<run_timestamp>/
experiments/<dataset>/with_rxn_class/<run_timestamp>/
```

Prediction files are written under the matching experiment directory.

## Legacy Script Equivalents

The historical scripts are still available:

```bash
python preprocess.py --mode train --dataset uspto_50k
python preprocess.py --mode valid --dataset uspto_50k
python preprocess.py --mode test --dataset uspto_50k

python prepare_data.py --dataset uspto_50k
python prepare_data.py --dataset uspto_50k --use_rxn_class

python train.py --dataset uspto_50k
python train.py --dataset uspto_50k --use_rxn_class

python eval.py --dataset uspto_50k
python eval.py --dataset uspto_50k --use_rxn_class
python eval-full.py --dataset uspto_full
python eval-rtacc.py --dataset uspto_50k
```

## Tests

Run the test suite with:

```bash
python -m pytest tests -q
```

## More Documentation

Contextual FG implementation notes are in:

```text
docs/contextual_fg.md
```
