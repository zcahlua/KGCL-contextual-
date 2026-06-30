# KGCL

KGCL is a retrosynthesis model based on molecular graph editing.

Paper title:

```text
KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction
Based on Molecular Graph Editing
```

This repository keeps the original KGCL behavior available and adds an optional
contextual functional-group mode. The contextual mode changes the
functional-group representation and does not expand the graph-edit action
space.

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

Run the default baseline KGCL workflow on USPTO-50K without reaction class:

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test

kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode valid
kgcl-prepare-data --dataset uspto_50k --mode test

kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

The `kgcl-*` package commands are preferred for new runs. The legacy top-level
scripts still work and are listed below.

## Reaction-class conditioning

The default setting is reaction-class-unknown: `--use_rxn_class` is off unless
you pass it explicitly. To run reaction-class-known KGCL, pass
`--use_rxn_class` consistently to data preparation, training, and evaluation.

The code uses the flag to select separate paths:

```text
data/<dataset>/<split>/without_rxn_class/
data/<dataset>/<split>/with_rxn_class/
experiments/<dataset>/without_rxn_class/<experiment>/
experiments/<dataset>/with_rxn_class/<experiment>/
```

USPTO-50K supports both reaction-class-unknown and reaction-class-known runs.
The USPTO-FULL commands in this README use the reaction-class-unknown setting,
which matches the bundled `experiments/uspto_full/without_rxn_class/BEST/`
checkpoint layout.

Without reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode valid
kgcl-prepare-data --dataset uspto_50k --mode test
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

With reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class
kgcl-train --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
```

## Contextual functional-group model

Baseline KGCL uses functional-group knowledge from a chemical knowledge graph:
SMARTS matches are reduced to functional-group types, KG embeddings are looked
up, and attention injects those embeddings into atom representations before the
D-MPNN graph encoder.

The contextual variant is enabled with `--fg_mode contextual`. It keeps matched
functional-group instances in the current molecule, records local k-hop
molecular context around each match, encodes that context with a trainable
functional-group graph encoder, fuses it with the KG embedding, and applies
atom-to-functional-group attention using membership and distance features.

The default model is baseline KGCL because `--fg_mode` defaults to `legacy` for
prepare-data and train. Evaluation loads the functional-group architecture saved
in the checkpoint by default. Passing `--fg_mode contextual` during evaluation is
a compatibility check against the checkpoint; it does not convert a baseline
checkpoint into a contextual checkpoint.

Prepared data is not freely interchangeable across functional-group modes.
Legacy mode writes directly under `with_rxn_class/` or `without_rxn_class/`.
Contextual mode writes under a mode-specific subdirectory such as
`contextual_fg_r1/` and includes `fg_metadata.json`. Training checks this
metadata and raises if the prepared data does not match the requested
functional-group configuration.

Contextual KGCL without reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode valid --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode test --fg_mode contextual
kgcl-train --dataset uspto_50k --fg_mode contextual
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual
```

Contextual KGCL with reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class --fg_mode contextual
kgcl-train --dataset uspto_50k --use_rxn_class --fg_mode contextual
kgcl-eval-50k --dataset uspto_50k --use_rxn_class --fg_mode contextual
```

Common contextual options:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual --fg_context_radius 1
kgcl-train --dataset uspto_50k --fg_mode contextual --fg_context_radius 1
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual --fg_context_radius 1
```

Additional functional-group options are available in prepare-data and train:

- `--fg_context_radius`
- `--fg_hidden_size`
- `--fg_layers`
- `--fg_dropout`
- `--fg_attn_dim`
- `--fg_max_dist`
- `--fg_max_matches_per_pattern`
- `--fg_use_kg_fusion` / `--no_fg_use_kg_fusion`
- `--fg_use_membership_bias` / `--no_fg_use_membership_bias`
- `--fg_use_distance_bias` / `--no_fg_use_distance_bias`
- `--fg_null_token` / `--no_fg_null_token`
- `--fg_freeze_kg_projection`

`--fg_mode contextual_fg` is accepted as an alias for `--fg_mode contextual`.
`--fg_mode none` disables functional-group fusion for ablation.

## Top-k evaluation

Evaluation uses `BeamSearch` with `step_beam_size=10`. The `--beam_size` option
controls how many final candidates are retained for top-k reporting.

USPTO-50K evaluation reports cumulative top-k exact-match accuracy for
`k = 1, 3, 5, 10, 50`. It also reports MaxFrag top-k accuracy, where the largest
predicted reactant fragment is compared with the largest ground-truth fragment.
The default USPTO-50K beam size is 50.

```bash
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --beam_size 50
```

USPTO-FULL evaluation reports cumulative top-k exact-match accuracy for
`k = 1, 3, 5, 10`. The default USPTO-FULL beam size is 10.

```bash
kgcl-eval-full --dataset uspto_full
kgcl-eval-full --dataset uspto_full --beam_size 10
```

Round-trip evaluation is available for USPTO-50K. It reports exact-match and
round-trip top-k accuracy for `k = 1, 3, 5, 10, 50`. The script expects the
forward-model prediction file under the selected experiment directory, for
example `forward_predictions_50k_top50.txt` under
`experiments/uspto_50k/without_rxn_class/BEST/`.

```bash
kgcl-eval-roundtrip --dataset uspto_50k
kgcl-eval-roundtrip --dataset uspto_50k --beam_size 50
```

## Metrics

Top-k exact-match accuracy checks whether the canonicalized ground-truth
reactant set appears among the top-k predicted reactant candidates.

MaxFrag accuracy checks whether the largest predicted reactant fragment matches
the largest ground-truth reactant fragment. This is implemented by the USPTO-50K
evaluation script.

Round-trip accuracy checks whether a forward-synthesis model reconstructs the
product from predicted reactants. This is implemented by the round-trip
evaluation script and depends on the external forward-model prediction file.

## Option table

| Option | Default | Used by | Effect |
| --- | ---: | --- | --- |
| `--use_rxn_class` | off | prepare/train/eval | Enables reaction-class conditioning and switches data/checkpoint paths between `without_rxn_class` and `with_rxn_class`. |
| `--fg_mode legacy` | `legacy` | prepare/train | Uses original KGCL functional-group type lookup and atom attention. |
| `--fg_mode contextual` | off | prepare/train/eval | Enables contextual functional-group KGCL. Evaluation uses the checkpoint architecture unless this flag is supplied as a compatibility check. |
| `--fg_context_radius` | `1` | prepare/train/eval | Sets the k-hop atom context around each matched functional-group instance. |
| `--beam_size` | `50` for 50K and round-trip; `10` for FULL | eval | Controls beam-search candidate generation for top-k evaluation. |
| `--experiments` | `BEST` | eval | Selects the experiment/checkpoint directory under the reaction-class path. |
| `--root_dir` | `.` | package CLI | Root directory containing `data/` and `experiments/`. |
| `--allow_architecture_override` | off | eval | Allows eval-time FG architecture flags to override checkpoint config intentionally. |

## Common recipes

### Baseline KGCL without reaction class

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode valid
kgcl-prepare-data --dataset uspto_50k --mode test
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

### Baseline KGCL with reaction class

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class
kgcl-train --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
```

### Contextual KGCL without reaction class

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test
kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode valid --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode test --fg_mode contextual
kgcl-train --dataset uspto_50k --fg_mode contextual
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual
```

### Contextual KGCL with reaction class

```bash
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode valid
kgcl-preprocess --dataset uspto_50k --mode test
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class --fg_mode contextual
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class --fg_mode contextual
kgcl-train --dataset uspto_50k --use_rxn_class --fg_mode contextual
kgcl-eval-50k --dataset uspto_50k --use_rxn_class --fg_mode contextual
```

### USPTO-FULL evaluation

```bash
kgcl-eval-full --dataset uspto_full
kgcl-eval-full --dataset uspto_full --beam_size 10
```

### Round-trip evaluation

```bash
kgcl-eval-roundtrip --dataset uspto_50k
kgcl-eval-roundtrip --dataset uspto_50k --beam_size 50
```

## Notes and pitfalls

Run preprocessing before prepare-data, and run prepare-data for `train`,
`valid`, and `test` before training and evaluation.

Use the same `--use_rxn_class` setting for prepare-data, train, and eval. The
flag changes feature dimensions and moves data/checkpoints between
`with_rxn_class` and `without_rxn_class` paths.

Use the same baseline/contextual functional-group setting for prepare-data,
train, and eval. Contextual prepared data includes metadata that training checks
against the requested FG architecture. Baseline and contextual prepared data may
not be interchangeable.

Evaluation should use a checkpoint trained with the intended model setting.
Passing `--fg_mode contextual` at eval time does not upgrade a baseline
checkpoint; it only requests that evaluation verify or intentionally override
the checkpoint architecture.

Do not compare USPTO-50K and USPTO-FULL top-k numbers directly without noting
that the datasets differ. Do not compare reaction-class-known and
reaction-class-unknown results as if they were the same setting.

Do not claim contextual KGCL accuracy numbers unless the repository includes
verified logs, checkpoints, or paper results for the exact setting.

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

## Package commands

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
kgcl-prepare-data --dataset uspto_50k --mode train --fg_mode contextual
```

Train:

```bash
kgcl-train --dataset uspto_50k
kgcl-train --dataset uspto_50k --use_rxn_class
kgcl-train --dataset uspto_50k --fg_mode contextual
```

Evaluate:

```bash
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --fg_mode contextual
kgcl-eval-full --dataset uspto_full
kgcl-eval-roundtrip --dataset uspto_50k
```

Prediction files are written under the matching experiment directory.

## Legacy script equivalents

The historical scripts are still available:

```bash
python preprocess.py --mode train --dataset uspto_50k
python preprocess.py --mode valid --dataset uspto_50k
python preprocess.py --mode test --dataset uspto_50k

python prepare_data.py --dataset uspto_50k --mode train
python prepare_data.py --dataset uspto_50k --mode train --use_rxn_class
python prepare_data.py --dataset uspto_50k --mode train --fg_mode contextual

python train.py --dataset uspto_50k
python train.py --dataset uspto_50k --use_rxn_class
python train.py --dataset uspto_50k --fg_mode contextual

python eval.py --dataset uspto_50k
python eval.py --dataset uspto_50k --use_rxn_class
python eval.py --dataset uspto_50k --fg_mode contextual
python eval-full.py --dataset uspto_full
python eval-rtacc.py --dataset uspto_50k
```

## Contextual FG diagnostics

Graph preparation is quiet by default. For local profiling, set:

```bash
export KGCL_PROFILE_GRAPH_BUILD=1
export KGCL_PROFILE_CONTEXTUAL_FG=1
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

## Tests

Run the test suite with:

```bash
python -m pytest tests -q
```

## More documentation

Contextual FG implementation notes are in:

```text
docs/contextual_fg.md
```
