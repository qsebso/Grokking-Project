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
| `--data_seed`         | `42`                  | Seeds **both** (1) the Python RNG that shuffles all `(a,b)` pairs and splits train/val, and (2) a dedicated PyTorch `Generator` for **which** train rows get corrupted and **how** (for all asymmetric modes). Same `data_seed` + same other data flags ⇒ identical noisy training labels across runs and across `main.py` / `train_category` / `train_category_factor`. Model init uses a separate **`model_seed`** (default `42`) in `TrainConfig`. |
| `--input_format`      | `a_op_b_eq`           | Token layout (see below)                                                                                                                                                                                                             |
| `--rule_count`        | `1`                   | `1`: labels in `0..p-1`. `n>1`: disjoint bands `0..p-1`, `p..2p-1`, …; `**n` must match the op’s branch count** (e.g. `add_or_mul` → 2, 3-way ops → 3). See `OPERATION_RULE_INFO` in `data/dataset.py`.                              |
| `--label_noise`       | `0.0`                 | Fraction of **training** rows to corrupt with asymmetric noise (`floor(n_train * value)` rows). Meaning of “corrupt” is set by **`--noise_mode`** (below). Val labels are always clean.                                              |
| `--noise`             | none                  | If set, **overrides** `--label_noise` (asymmetric fraction).                                                                                                                                                                         |
| `--noise_sym`         | `0.0`                 | **Symmetric** train noise: swap labels in random disjoint pairs (same seeded generator after asymmetric step). Asymmetric runs first, then symmetric. **Not** supported for S5 in `main.py`.                                           |


### Train label noise: modes, fixed targets, and seeding

All logic lives in `data/dataset.py` (`apply_train_label_corruption`, `NOISE_MODE_CHOICES`).

**Fraction:** `--noise` / `--label_noise` sets the asymmetric fraction; `--noise_sym` sets symmetric pair swaps. Only the **training** split is modified.

**Mode:** `--noise_mode` chooses *how* corrupted labels are chosen, or use exactly one shorthand flag:

| `--noise_mode` / shorthand flag | When it applies | Behavior |
| ------------------------------- | --------------- | -------- |
| `random_wrong_c` (default) / `--random_wrong_c` | Any op, any `rule_count` | Random **incorrect** class uniform over all wrong labels (same label space as the task). |
| `fixed_wrong_c` / `--fixed_wrong_c` | `rule_count == 1` | Corrupted rows → `--noise_fixed_target` (mod `num_classes`); if that equals the true label, use `--noise_fixed_backup` or the next class. |
| `shifted_wrong_c` / `--shifted_wrong_c` | `rule_count == 1` | Corrupted rows → `(y + k) mod num_classes` with smallest `k ≥ 1` that changes the label. |
| `other_rule_c` / `--other_rule_c` | `rule_count == 2`, op in add/mul family (see code) | On add branch, label becomes **mul** output; on mul branch, label becomes **add** output (same branch band). For **categorical** runs, requires `--label_mode c`. |
| `fixed_wrong_c_cross_rule` / `--fixed_wrong_c_cross_rule` | `rule_count ≥ 2` | Like fixed wrong, but `--noise_fixed_target` is a **global** class in `0 .. rule_count*p - 1`. |

**Fixed-target knobs:** `--noise_fixed_target` (default `5`), `--noise_fixed_backup` (optional; default picks another class if target collides with the true label).

**Seeding (reproducibility):** Fix **`--data_seed`** to compare different models or hyperparameters on the **same** train/val split and the **same** corruption pattern (subset of noisy indices + random wrong labels in `random_wrong_c`, etc.). Changing only `--model_seed` (via `TrainConfig` / future CLI if added) changes initialization but not the data or noise.

**S5:** Only `random_wrong_c` is allowed for asymmetric noise on permutation tasks.

**Outputs:** JSON `summary` includes `noise_mode`, `noise_fixed_target`, and optional `noise_fixed_backup` when relevant; result filenames add fragments such as `_nasy0p1` and, for non-default modes, `_nm…` / `_ft…` (see `experiments/train.py` → `noise_fname_suffix`).

**Categorical / factor only** (also on `train_category` and `train_category_factor`): targets are built with `make_category_dataset` (`label_mode`, `label_mod`, `rule_count`). Structured modes such as `other_rule_c` may require `label_mode=c` (enforced at dataset build time).

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

# Same split + same noise as above, but pin seed explicitly (default is 42 if omitted):
python main.py --operation add --noise 0.1 --data_seed 42 --shifted_wrong_c
python main.py --operation add_or_mul --rule_count 2 --noise 0.1 --data_seed 42 --other_rule_c
python -m experiments.train_category_factor --operation add_or_mul --rule_count 2 --noise 0.1 --data_seed 42 --fixed_wrong_c_cross_rule --noise_fixed_target 100
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

Default `**--results_dir**` is `**results**` (same as `main.py` / `train_category`; use a custom path if you want a separate folder).

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
- `**train_category_factor`:** `results/` by default; filenames include `_factor_<mode>_rc<n>` (and optional `_mtconvexified` when not standard).

Each JSON includes `**summary`** (hyperparameters, memo/grok epochs, `**label_noise**` / `**label_noise_sym**`, `**noise_mode**` / fixed-target fields when applicable, `**data_seed**`) plus logged curves.

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
results/                         # Default JSON output (all entry points; optional legacy results_factor/)
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