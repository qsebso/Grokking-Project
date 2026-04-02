# Grokking Experiments

Modular arithmetic and related tasks on a small decoder-only transformer. You can run **token prediction** (`main.py`), **categorical classification** (`train_category`), or **factored rule + output heads** (`train_category_factor`). Shared data and models live under `data/` and `models/`; training loops under `experiments/`.

---

## Which script to run


| Entry point                                       | Task                                              | Target                                                               | Typical use                                           |
| ------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------- |
| `**python main.py`**                              | Next-token / vocab logits                         | Answer token (or disjoint bands if `rule_count > 1`)                 | Paper-style grokking, multi-rule via single softmax   |
| `**python -m experiments.train_category**`        | Cross-entropy on **class id**                     | `label_mode` (e.g. `c`, `c_parity`, `b_parity`, …)                   | Probes that depend on true output vs input-only stats |
| `**python -m experiments.train_category_factor`** | Same disjoint labels as `c` with `rule_count ≥ 2` | **joint**: rule head + local `c` head; or **rule_only** / **c_only** | Ablations: separate rule vs output prediction         |


All commands below assume your working directory is `**Grokking-Project`**.

---

## Shared options (all three runners)

These names match `TrainConfig` in `experiments/train.py` unless noted.

### Data


| Flag                  | Default               | Meaning                                                                                                                                                                                                                              |
| --------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--operation`         | `add` (main) / varies | Operation key in `data/dataset.py` → `OPERATIONS`                                                                                                                                                                                    |
| `--p`                 | `97`                  | Modulus (ignored for S5 ops)                                                                                                                                                                                                         |
| `--train_frac`        | `0.5`                 | Fraction of shuffled pairs → train; rest → val                                                                                                                                                                                       |
| `--max_train_samples` | none                  | Hard cap on train size after split                                                                                                                                                                                                   |
| `--data_seed`         | `42`                  | Shuffles/split for dataset construction                                                                                                                                                                                              |
| `--input_format`      | `a_op_b_eq`           | Token layout (see below)                                                                                                                                                                                                             |
| `--rule_count`        | `1`                   | `1`: labels in `0..p-1`. `n>1`: disjoint bands `0..p-1`, `p..2p-1`, …; `**n` must match the op’s branch count** (e.g. `add_or_mul` → 2, 3-way ops → 3). See `OPERATION_RULE_INFO` in `data/dataset.py`.                              |
| `--label_noise`       | `0.0`                 | **Asymmetric** train noise: replace a fraction of train labels with uniform random **valid** targets (implementation differs slightly: main uses answer range; categorical uses `0..num_classes-1`).                                 |
| `--noise`             | none                  | If set, **overrides** `--label_noise` (asymmetric).                                                                                                                                                                                  |
| `--noise_sym`         | `0.0`                 | **Symmetric** train noise: swap labels in random disjoint pairs (no change in marginal label distribution). If both asymmetric and symmetric are > 0, asymmetric is applied first, then symmetric. **Not** used for S5 in `main.py`. |


**Categorical / factor only** (also on `train_category` and `train_category_factor`): targets are built with `make_category_dataset` (`label_mode`, `label_mod`, `rule_count`).

### Input formats (`--input_format`)


| Value               | Sequence (integers mod p)                         |
| ------------------- | ------------------------------------------------- |
| `a_op_b_eq`         | `[a, op_tok, b, =]` (paper default)               |
| `a_b_eq`            | `[a, b, =]`                                       |
| `a_op_b_eq_rule`    | `[rule_tok, a, op_tok, b, =]` (extra vocab token) |
| `a_op_b_eq_bparity` | `[a, op_tok, b, b%2, =]`                          |
| `a_op_bparity_eq`   | `[a, op_tok, b%2, =]` (no full `b`)               |


### Optimizer and model


| Flag                | Default | Meaning                                  |
| ------------------- | ------- | ---------------------------------------- |
| `--lr`              | `1e-3`  | AdamW learning rate                      |
| `--weight_decay`    | `1.0`   | AdamW weight decay                       |
| `--d_model`         | `128`   | Hidden size                              |
| `--nhead`           | `4`     | Attention heads                          |
| `--num_layers`      | `2`     | Transformer layers                       |
| `--dim_feedforward` | auto    | FFN inner dim (`4 * d_model` if omitted) |


### Training schedule


| Flag              | Default               | Meaning                                                                                                                                                          |
| ----------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--num_epochs`    | `5000` (main default) | Full training length                                                                                                                                             |
| `--batch_size`    | none                  | Omit or larger than train size → **full-batch**; else DataLoader                                                                                                 |
| `--log_every`     | `50`                  | Log metrics every N epochs                                                                                                                                       |
| `--branch_metric` | `auto`                | How **branch** accuracies are split: `b_parity`, `a_ge_b`, `a_gt_b`, or `auto` (from operation). Logged as train/val “odd/even” lists in JSON for compatibility. |


### Outputs


| Flag                | Meaning                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `--results_dir`     | JSON directory (`results`, `results_factor`, or custom)                                     |
| `--save_checkpoint` | Optional `.pt` path: saves `model_state_dict` + `config` after training (for PCA / reload). |
| `--quiet`           | Less console output                                                                         |


**Result filenames** encode key settings (operation, `p`, `wd`, `lr`, `d_model`, `num_layers`, `train_frac`, optional `max_train_samples`, `input_format`, `rule_count`, **noise** tags like `_nasy0p1_sym0p05`, etc.).

---

## `main.py` — token prediction

- **Dataset:** `make_dataset` → softmax over `vocab_size`, or `rule_count * p` when `rule_count > 1`.
- **Model:** `models/transformer.py` → `TransformerModel`.
- **Sweeps:** `--sweep PARAM v1 v2 …` or `--grid1 … --grid2 …` (every combination).

Examples:

```bash
python main.py --weight_decay 0.1 --num_epochs 10000
python main.py --sweep weight_decay 0.0 0.1 1.0
python main.py --operation add_or_mul --rule_count 2 --input_format a_op_b_eq_bparity
python main.py --noise 0.1 --noise_sym 0.0 --save_checkpoint checkpoints/run.pt
```

---

## `experiments.train_category` — categorical targets

- **Dataset:** `make_category_dataset` — same token sequences as `main`, but targets are **class indices** (`num_classes` depends on `label_mode` and `rule_count`).
- **Model:** `TransformerModel` with `num_logits = num_classes`.

### Extra flags


| Flag           | Meaning                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `--label_mode` | `c`, `c_parity`, `b_parity`, `a_parity`, `c_mod3`, `a_plus_b_mod3`, `c_mod`, `a_plus_b_mod`                  |
| `--label_mod`  | Modulus `k` for `c_mod` / `a_plus_b_mod` (ignored otherwise; default `0` in CLI — use e.g. `11` when needed) |


`**label_mode=c`** with `**rule_count > 1**`: disjoint global class `0 .. rule_count*p - 1` (same encoding as multi-rule token task).

Examples:

```bash
python -m experiments.train_category --operation add_or_mul --label_mode c_parity
python -m experiments.train_category --operation add_or_mul --rule_count 2 --label_mode c --branch_metric b_parity
python -m experiments.train_category --noise 0.1 --label_mode c --rule_count 2
```

**Sweeps:** `--sweep` / `--grid1` `--grid2` like `main.py` (fields must exist on `TrainCategoryConfig`).

---

## `experiments.train_category_factor` — factored heads (needs `rule_count ≥ 2`)

- **Dataset:** always `**label_mode=c`** with disjoint labels `y ∈ {0,…, rule_count*p - 1}` (same as categorical `c` + multi-rule).
- **Modes (`--factor_mode`):**
  - `**joint`**: `FactoredTransformer` — rule logits + `c` logits; loss = CE_rule + CE_c; logged **joint** accuracy = both correct.
  - `**rule_only`**: single head, target `y // p`.
  - `**c_only**`: single head, target `y % p`.

Default `**--results_dir**` is `**results_factor**` (kept separate from `results/`).

### Extra flags


| Flag                                                                                | Meaning                                                                            |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `--factor_mode`                                                                     | `joint` | `rule_only` | `c_only`                                                   |
| `--auto_pca`                                                                        | After training, run `analysis/pca_hidden_states.py` (requires `--save_checkpoint`) |
| `--pca_split`, `--pca_max_samples`, `--pca_layer`, `--pca_pool`, `--pca_output_dir` | Passed through to PCA script                                                       |


Example (joint + checkpoint + PCA):

```bash
python -m experiments.train_category_factor --factor_mode joint --operation add_or_mul --rule_count 2 \
  --input_format a_op_b_eq_bparity --num_epochs 10000 \
  --save_checkpoint checkpoints/factor_joint.pt --auto_pca --pca_output_dir analysis/out/pca_run
```

---

## Analysis utilities (from repo root)


| Script                            | Role                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis/analysis.py`            | Scan JSONs (`categorical` and `categorical_factor` with `label_mode=c`); CSVs, plots, per-run `summary.txt` (includes **label noise** fields when present).                                       |
| `analysis/difficulty_analysis.py` | Symmetry / difficulty views on selected runs.                                                                                                                                                     |
| `analysis/pca_hidden_states.py`   | PCA on hidden states; `**--checkpoint`** must point to a file saved with `**--save_checkpoint**`. Recreates dataset from checkpoint `config` (including **noise** and categorical vs token path). |
| `analysis/verify_rule_labels.py`  | Sanity-check rule ids vs inputs; supports `--noise` / `--noise_sym`.                                                                                                                              |


---

## Where results go

- `**main.py`:** `results/` (or `--results_dir`).
- `**train_category`:** same default; filenames include `_cat_<label_mode>` and optional `_rc<n>` for `rule_count > 1`.
- `**train_category_factor`:** `results_factor/` by default; filenames include `_factor_<mode>_rc<n>`.

Each JSON includes `**summary`** (hyperparameters, memo/grok epochs, `**label_noise**` / `**label_noise_sym**` when applicable) plus logged curves.

---

## Adding a new operation

Edit `**data/dataset.py**`:

1. Implement `op_my_rule(a, b, p)`.
2. Register in `**OPERATIONS**`: `(fn, False, domain_fn)` for mod-p, or `(fn, True, domain_fn)` for S5.
3. For **multi-rule** disjoint bands, add `**OPERATION_RULE_INFO`** if the op has multiple branches.

Then run any entry point with `--operation my_rule`.

---

## File map

```
main.py                          # Token-prediction experiments + sweeps
experiments/train.py             # TrainConfig, train loop, noise filename helper
experiments/train_category.py    # Categorical experiments
experiments/train_category_factor.py  # Factored heads (joint / rule_only / c_only)
data/dataset.py                  # Operations, datasets, label noise, rule ids
models/transformer.py            # Single-head transformer
models/transformer_factor.py     # Shared trunk + rule + c heads
analysis/                        # analysis.py, pca_hidden_states.py, verify_rule_labels.py, …
results/                         # Default JSON output (main + train_category)
results_factor/                  # Default JSON output (train_category_factor)
```

---

## Operations reference (subset)

Full list: `sorted(OPERATIONS.keys())` or `data/dataset.py`.


| Name                                    | Idea                                                    |
| --------------------------------------- | ------------------------------------------------------- |
| `add`, `sub`, `mul`, `div`, …           | Binary ops mod `p`                                      |
| `add_or_mul`, `4way_add_sub_mul_div`, … | Multi-branch rules; use matching `**rule_count**`       |
| `s5_mul`, `s5_conj`, `s5_sandwich`      | Permutation group S₅ (no categorical / factor pipeline) |


Default modulus `**p = 97**` unless you set `--p`.