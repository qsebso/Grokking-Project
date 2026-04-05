# Grokking Experiments

Modular arithmetic and related tasks on small transformer models. Entry points:

- **Token prediction:** `main.py`
- **Categorical classification:** `experiments.train_category`
- **Factored rule + local output heads:** `experiments.train_category_factor` (includes **standard**, **convexified**, and **routed modular** architectures)

Shared data and models live under `data/` and `models/`; training loops under `experiments/`.

All example commands assume your working directory is **`Grokking-Project`** (the repo root).

---

## Which script to run

| Entry point | Task | Target | Typical use |
|-------------|------|--------|-------------|
| `python main.py` | Next-token / vocab logits | Answer token (or disjoint bands if `rule_count > 1`) | Paper-style grokking, multi-rule via single softmax |
| `python -m experiments.train_category` | Cross-entropy on **class id** | `label_mode` (e.g. `c`, `c_parity`, …) | Probes on true output vs input-only stats |
| `python -m experiments.train_category_factor` | Same disjoint labels as `c` with `rule_count ≥ 2` | **joint:** rule head + `c` head(s); **rule_only** / **c_only** single heads | Ablations: factored heads, shared MLP `c`-head, routed experts |

---

## Repository file map

| Path | Role |
|------|------|
| `main.py` | Token-prediction training, sweeps/grids |
| `experiments/train.py` | `TrainConfig`, `TrainResult`, `MODEL_TYPE_CHOICES`, `build_sequence_model`, `build_factored_model`, `train()`, `noise_fname_suffix`, branch masks for logging |
| `experiments/train_category.py` | Categorical runs; builds `TrainCategoryConfig`, uses `build_sequence_model` |
| `experiments/train_category_factor.py` | Factored runs; `TrainFactorConfig`, joint/rule_only/c_only, **routed** metadata, `routed_analysis` JSON, CLI for all factor-specific flags |
| `data/dataset.py` | `OPERATIONS`, `OPERATION_RULE_INFO`, `make_dataset`, `make_category_dataset`, label noise, `resolve_rule_id`, vocab/seq layout |
| `models/transformer.py` | `TransformerModel` — single softmax head (used by `main`, categorical, factor rule_only/c_only) |
| `models/transformer_factor.py` | `FactoredTransformer` — shared trunk, `head_rule`, configurable **shared** `head_c` (linear or MLP via `shared_c_head_layers`) |
| `models/transformer_convex.py` | Convexified attention trunk; `ConvexifiedTransformer`, `ConvexifiedFactoredTransformer` |
| `models/transformer_routed.py` | `RoutedModularTransformer` — shared trunk, `rule_head`, `head_router`, `c_heads` (count = `c_head_count`), soft/hard routing |
| `plots/plot_results.py` | Plot saved JSON curves; auto **routed analysis** figure when `routed_analysis` is present |
| `analysis/analysis.py` | Scan JSON trees for categorical/factor runs; CSV + per-run folders; includes model/routing/c-head metadata columns |
| `analysis/pca_hidden_states.py` | PCA on hidden states from `--save_checkpoint` checkpoints |
| `analysis/verify_rule_labels.py` | Sanity-check rule ids vs inputs |
| `analysis/difficulty_analysis.py` | Symmetry / difficulty diagnostics on selected runs |
| `train_spreadsheet/build_spreadsheet.py` | Build CSV from JSON results |
| `organized_results/summary_maker.py` | Summarize a results directory |
| `data/plot_symmetrical_dataset_maps.py` | Dataset visualization helpers |
| `results/` | Default JSON output (or `--results_dir`) |

---

## Shared options (all runners)

Names match `TrainConfig` in `experiments/train.py` unless noted.

### Data

| Flag | Default | Meaning |
|------|---------|---------|
| `--operation` | varies | Key in `data/dataset.py` → `OPERATIONS` |
| `--p` | `97` | Modulus (ignored for S5 ops) |
| `--train_frac` | `0.5` | Fraction of shuffled pairs → train |
| `--max_train_samples` | none | Cap train size after split |
| `--data_seed` | `42` | Shuffles/split + noise generator seed (train/val + corruption pattern) |
| `--input_format` | `a_op_b_eq` | Token layout (see below) |
| `--rule_count` | `1` | For multi-rule ops, must match branch count (see `OPERATION_RULE_INFO` in `dataset.py`) |
| `--label_noise` / `--noise` / `--noise_sym` | | Asymmetric/symmetric train label noise (train split only; val stays clean) |
| `--noise_mode` | `random_wrong_c` | How corrupted labels are chosen; see table below and `NOISE_MODE_CHOICES` in `data/dataset.py` |

**Train label noise (detail).** Logic: `apply_train_label_corruption` in `data/dataset.py`. Fraction: `--noise` overrides `--label_noise`; `--noise_sym` adds symmetric pair swaps after asymmetric corruption.

| `--noise_mode` / shorthand | When it applies | Behavior |
|----------------------------|-----------------|----------|
| `random_wrong_c` (default) / `--random_wrong_c` | Any op, any `rule_count` | Random incorrect class uniform over wrong labels. |
| `fixed_wrong_c` / `--fixed_wrong_c` | `rule_count == 1` | Corrupted rows → `--noise_fixed_target` (mod `num_classes`); backup if collision. |
| `shifted_wrong_c` / `--shifted_wrong_c` | `rule_count == 1` | `(y + k) mod num_classes` with smallest `k ≥ 1` that changes the label. |
| `other_rule_c` / `--other_rule_c` | `rule_count == 2`, add/mul family | On add branch, label becomes mul output and vice versa (same band). Categorical/factor often needs `label_mode=c`. |
| `fixed_wrong_c_cross_rule` / `--fixed_wrong_c_cross_rule` | `rule_count ≥ 2` | Fixed wrong with `--noise_fixed_target` as **global** class in `0 .. rule_count·p − 1`. |

**Knobs:** `--noise_fixed_target` (default `5`), `--noise_fixed_backup` (optional). **Seeding:** same `--data_seed` ⇒ same split and same corruption pattern across `main` / `train_category` / `train_category_factor`. **`model_seed`** (default `42` in factor config) changes init only.

**S5:** only `random_wrong_c` for asymmetric noise on permutation tasks.

JSON `summary` includes `noise_mode`, fixed-target fields when relevant; filenames use `noise_fname_suffix` in `experiments/train.py` (e.g. `_nasy0p1`, `_nm…`).

### Input formats (`--input_format`)

| Value | Sequence (integers mod `p`) |
|-------|-----------------------------|
| `a_op_b_eq` | `[a, op_tok, b, =]` |
| `a_b_eq` | `[a, b, =]` |
| `a_op_b_eq_rule` | `[rule_tok, a, op_tok, b, =]` |
| `a_op_b_eq_bparity` | `[a, op_tok, b, b%2, =]` |
| `a_op_bparity_eq` | `[a, op_tok, b%2, =]` |

### Optimizer and trunk

| Flag | Default | Meaning |
|------|---------|---------|
| `--lr` | `1e-3` | AdamW LR |
| `--weight_decay` | `1.0` | AdamW weight decay |
| `--d_model` | `128` | Hidden size |
| `--nhead` | `4` | Attention heads |
| `--num_layers` | `2` | Transformer layers |
| `--dim_feedforward` | auto | `4 * d_model` if omitted |

### Schedule and logging

| Flag | Default | Meaning |
|------|---------|---------|
| `--num_epochs` | `5000` (varies by script) | Training length |
| `--batch_size` | none | Full-batch if omitted or larger than train |
| `--log_every` | `50` | Log every N epochs |
| `--branch_metric` | `auto` | `b_parity`, `a_ge_b`, `a_gt_b`, or `auto` from operation |

### Outputs

| Flag | Meaning |
|------|---------|
| `--results_dir` | JSON directory (default `results`) |
| `--save_checkpoint` | Optional `.pt` with `state_dict` + `config` |
| `--quiet` | Less console output |

**Noise in filenames:** `experiments/train.py` → `noise_fname_suffix` adds fragments like `_nasy0p1` when noise is non-zero.

---

## `experiments.train_category_factor` — factored heads (`rule_count ≥ 2`)

### Dataset and modes

- Always **`label_mode=c`** with disjoint labels `y ∈ {0, …, rule_count·p − 1}` (same encoding as categorical `c` with multi-rule).
- **`--factor_mode`**
  - **`joint`:** two outputs — rule logits + local `c` logits; loss = CE(rule) + CE(c); **joint accuracy** = both correct.
  - **`rule_only`:** single head, target `y // p`.
  - **`c_only`:** single head, target `y % p`.

### `--model_type` (joint mode uses `build_factored_model`)

| `model_type` | Class | Behavior |
|--------------|-------|----------|
| **`standard`** | `FactoredTransformer` | Shared trunk + linear rule head + **shared** `c` head (linear or MLP; see `--shared_c_head_layers`). |
| **`convexified`** | `ConvexifiedFactoredTransformer` | Convexified trunk + same two-head layout as before (no shared MLP flag on this path). |
| **`routed_modular`** | `RoutedModularTransformer` | Shared trunk + **per-expert** `c` heads (count = `--c_head_count`, default `rule_count`) + routing (see below). |

`MODEL_TYPE_CHOICES` is defined in `experiments/train.py`.

### Factor-only CLI flags

| Flag | Default | Applies to | Meaning |
|------|---------|------------|---------|
| `--factor_mode` | `joint` | all | `joint` \| `rule_only` \| `c_only` |
| `--model_type` | `standard` | all | `standard` \| `convexified` \| `routed_modular` |
| `--shared_c_head_layers` | `1` | **`standard` + `joint`** | `1` = single linear `c` head; `≥ 2` = shared MLP: `(Linear(d→d) + GELU)×(N−1)` then `Linear(d→p)`. No routing. |
| `--routing_mode` | `hard` | **`routed_modular`** | `hard` = argmax on rule logits picks one `c` head (requires `c_head_count == rule_count`). `soft` = weighted mixture of expert logits. |
| `--routed_c_head_layers` | `3` | **`routed_modular`** | Number of **Linear** layers inside **each** expert `c` head (1 = linear; default 3 = expand to `2d`, then `d`, then `p` with GELU). |
| `--c_head_count` | `None` → **`rule_count`** | **`routed_modular`** | How many expert `c` heads. If `< rule_count`, use **`--routing_mode soft`**; routing uses `head_router` (not rule argmax). If `== rule_count`, behavior matches the original soft path (route via rule logits when soft). |
| `--auto_pca` | off | all | After training, run `analysis/pca_hidden_states.py` (needs `--save_checkpoint`) |
| `--pca_*` | | | Passed to PCA script |

**Verbose logging** prints e.g. `Shared c-head: linear` / `Shared c-head: MLP (N layers)` for standard, and for routed: `c_head_count`, `rule_count`, `routing_mode`, `routed_c_head_layers`.

### Result JSON and filenames (`train_category_factor`)

Saved JSON includes:

- **`summary`:** usual hparams plus, when relevant:
  - `shared_c_head_layers` (always written; default `1` for non-standard paths is harmless)
  - `routing_mode`, `routed_c_head_layers`, `c_head_count` for `routed_modular`
- **`routed_analysis`** (only `routed_modular` + `joint`): aggregate train/val diagnostics — head usage, mean routing weights per true rule, per-head `c` accuracy by rule, entropy, confusion-style counts. Used by `plot_results.py` for the extra routing figure.

**Filename fragments** (in addition to operation, `p`, wd, lr, d, layers, `train_frac`, format, noise, `_factor_<mode>`, `_rc<n>`, `_ep<n>`):

- `_mt<model_type>` if not `standard` (e.g. `_mtconvexified`, `_mtrouted_modular`)
- `_rm<hard|soft>` for routed
- `_chc<N>` for routed when `c_head_count` is set (saved runs include it)
- `_chl<N>` for routed per-expert depth (`routed_c_head_layers`)
- `_schl<N>` for **standard** shared MLP when `shared_c_head_layers != 1`

### Example commands

**Shared linear baseline (standard joint):**
```bash
python -m experiments.train_category_factor --model_type standard --operation 4way_add_sub_mul_div --factor_mode joint --rule_count 4 --train_frac 0.5 --input_format a_op_b_eq_bparity --num_epochs 20000 --weight_decay 1
```

**Shared 2-layer MLP `c` head (no routing ablation):**
```bash
python -m experiments.train_category_factor --model_type standard --operation 4way_add_sub_mul_div --factor_mode joint --rule_count 4 --train_frac 0.5 --input_format a_op_b_eq_bparity --num_epochs 20000 --weight_decay 1 --shared_c_head_layers 2
```

**Routed modular, soft, 2-layer experts, default heads = rules:**
```bash
python -m experiments.train_category_factor --model_type routed_modular --routing_mode soft --operation 4way_add_sub_mul_div --factor_mode joint --rule_count 4 --train_frac 0.5 --input_format a_op_b_eq_bparity --num_epochs 20000 --weight_decay 1 --routed_c_head_layers 2
```

**Fewer experts than rules (e.g. 4 rules, 3 heads) — soft only:**
```bash
python -m experiments.train_category_factor --model_type routed_modular --routing_mode soft --operation 4way_add_sub_mul_div --factor_mode joint --rule_count 4 --c_head_count 3 --train_frac 0.5 --input_format a_op_b_eq_bparity --num_epochs 20000 --weight_decay 1 --routed_c_head_layers 2
```

**Small smoke test:**
```bash
python -m experiments.train_category_factor --model_type routed_modular --operation add_or_mul --factor_mode joint --rule_count 2 --train_frac 0.5 --input_format a_op_b_eq_bparity --num_epochs 20 --log_every 10 --p 17
```

**Joint + checkpoint + PCA:**
```bash
python -m experiments.train_category_factor --factor_mode joint --operation add_or_mul --rule_count 2 \
  --input_format a_op_b_eq_bparity --num_epochs 10000 \
  --save_checkpoint checkpoints/factor_joint.pt --auto_pca --pca_output_dir analysis/out/pca_run
```

### Routed model details (`transformer_routed.py`)

- **Outputs:** `rule_logits` shape `(B, rule_count)`, `c_logits` shape `(B, p)` — same interface as `FactoredTransformer` for the training loop.
- **`last_routing_info`:** after each forward — e.g. `predicted_rule`, `chosen_head`, `all_c_logits`, `routing_weights` (soft), `head_logits` when used, `c_head_count`, `c_head_layers`.
- **Hard routing:** index into `all_c_logits` with `argmax(rule_logits)`; only valid if `c_head_count == rule_count`.
- **Soft routing:** if `c_head_count == rule_count`, weights = `softmax(rule_logits)` (backward compatible). If `c_head_count != rule_count`, weights = `softmax(head_router(h))`.
- **Gradients:** hard routing does not backprop `c` loss through the discrete route; soft routing does.

---

## `plots/plot_results.py` — plotting saved JSON

Run from repo root.

```bash
python plots/plot_results.py
python plots/plot_results.py results/some_run.json
python plots/plot_results.py --mode single results/*.json
python plots/plot_results.py --mode sweep --sweep_param weight_decay results/*.json
python plots/plot_results.py --mode grid results/*.json
python plots/plot_results.py --metric loss --no_show
```

**Behavior:**

- Loads JSON from globs or `results/*.json` (fallback `results_factor/`).
- **Single-file mode:** writes accuracy+loss panel; if the JSON contains **`routed_analysis`**, also writes **`*_routing.png`** with head usage, entropy, mean weights by true rule, confusion-style heatmaps, per-head `c` accuracy by rule.
- Titles include task line with `routing`, `c_head_count`, `c_head_layers`, `shared_c_head_layers` when present in `summary`.
- Auto output names include `_schl*`, `_chc*`, `_chl*` segments when relevant so runs do not overwrite each other.

---

## Analysis utilities

| Script | Role |
|--------|------|
| `python analysis/analysis.py` | Walk directories (default `!previous_results` or pass `--roots results ...`). Finds `task` in `categorical` / `categorical_factor` with `label_mode=c`. Writes `analysis/all/categorical_c_summary.csv`, optional JSONL, per-run `runs/.../summary.txt`, plots. CSV columns include **`model_type`**, **`shared_c_head_layers`**, **`c_head_count`**, **`routing_mode`**, **`routed_c_head_layers`** when present in each JSON `summary`. |
| `python analysis/pca_hidden_states.py` | PCA from checkpoint; `--checkpoint` required. |
| `python analysis/verify_rule_labels.py` | Rule id checks. |
| `python analysis/difficulty_analysis.py` | Selected-run difficulty views. |

Example:

```bash
python analysis/analysis.py --roots results --no-plots
```

---

## `main.py` — token prediction

- **Dataset:** `make_dataset` → vocab softmax or `rule_count * p` when `rule_count > 1`.
- **Model:** `TransformerModel`.
- **Sweeps:** `--sweep` / `--grid1` / `--grid2`.

```bash
python main.py --weight_decay 0.1 --num_epochs 10000
python main.py --operation add_or_mul --rule_count 2 --input_format a_op_b_eq_bparity
```

---

## `experiments.train_category` — categorical

- **Dataset:** `make_category_dataset`.
- **Model:** `TransformerModel` with `num_logits = num_classes`.

Extra flags: `--label_mode`, `--label_mod` (for modular label modes). Sweeps like `main.py` on `TrainCategoryConfig` fields.

```bash
python -m experiments.train_category --operation add_or_mul --rule_count 2 --label_mode c --branch_metric b_parity
```

---

## Where results go

| Source | Default path | Filename hints |
|--------|--------------|----------------|
| `main.py` | `results/` | Encodes op, `p`, wd, lr, … |
| `train_category` | `results/` | `_cat_<label_mode>`, `_rc<n>` |
| `train_category_factor` | `results/` | `_factor_<mode>_rc<n>`, `_mt…`, `_rm…`, `_chc…`, `_chl…`, `_schl…` |

Each JSON has **`summary`** (hparams, memo/grok, noise fields, `data_seed`, **factor/routed fields** above) plus time series (`train_accs`, `val_accs`, losses, branch curves, etc.).

---

## Adding a new operation

Edit **`data/dataset.py`**:

1. Implement `op_my_rule(a, b, p)`.
2. Register in **`OPERATIONS`**: `(fn, False, domain_fn)` for mod-p, or `(fn, True, domain_fn)` for S5.
3. For multi-rule disjoint bands, add **`OPERATION_RULE_INFO`** with `(num_branches, rule_id_fn)`.

Then use `--operation my_rule` on any entry point.

---

## Operations reference (subset)

Full list: `sorted(OPERATIONS.keys())` in `data/dataset.py`.

| Name | Idea |
|------|------|
| `add`, `sub`, `mul`, `div`, … | Binary ops mod `p` |
| `add_or_mul`, `4way_add_sub_mul_div`, … | Multi-branch rules; **`rule_count` must match** |
| `s5_mul`, `s5_conj`, `s5_sandwich` | S₅ permutations (not the categorical/factor pipeline above) |

Default **`p = 97`** unless you pass `--p`.
