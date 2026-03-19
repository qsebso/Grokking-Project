# Grokking Experiments

Everything runs from `main.py`. You only ever touch two files:
- **`main.py`** — to run experiments
- **`data/dataset.py`** — to add new rules

---

## Running experiments

### One value
```bash
python main.py --weight_decay 0.1
```

### Multiple values at once (sweep)
```bash
python main.py --sweep weight_decay 0.0 0.1 1.0
```
This trains 3 separate models and prints a summary table at the end.

### Other things you can sweep or change the same way
```bash
python main.py --sweep lr 1e-4 3e-4 1e-3 3e-3

python main.py --sweep train_frac 0.3 0.5 0.7    # data split

python main.py --sweep d_model 128 256 512        # model width
python main.py --sweep num_layers 2 4 6           # model depth
python main.py --sweep nhead 4 8                  # attention heads

python main.py --sweep num_epochs 10000 50000     # training time
python main.py --sweep batch_size 64 256          # batch size (omit for full-batch)
```

### Absolute data size (hard cap on training examples)
```bash
python main.py --max_train_samples 1000
python main.py --sweep max_train_samples 500 1000 3000 5000
```
This caps the training set at exactly N examples regardless of `train_frac`.
You can combine both — `train_frac` cuts first, then the cap applies on top.

### Input representation / embedding format
```bash
# Default (matches the paper): [a, op, b, =]
python main.py --input_format a_op_b_eq

# No operator token: [a, b, =]
python main.py --input_format a_b_eq

# Explicit rule token prepended: [rule, a, op, b, =]
python main.py --input_format a_op_b_eq_rule

# Sweep all three
python main.py --sweep input_format a_op_b_eq a_b_eq a_op_b_eq_rule
```
The rule token format adds one extra token to the vocabulary and one extra
position to the input sequence. Useful for multi-rule experiments where you
want the model to see which rule is active.

### Change the operation
```bash
python main.py --operation sub
python main.py --sweep operation add sub div sq_sum s5_mul
```

### Combine anything
```bash
python main.py --operation sub --weight_decay 0.1 --lr 3e-4
python main.py --max_train_samples 1000 --input_format a_b_eq --num_epochs 10000
```

---

## All built-in operations

| Name | Formula |
|------|---------|
| `add` | x + y (mod p) |
| `sub` | x − y (mod p) |
| `div` | x / y (mod p), y ≠ 0 |
| `div_or_sub` | x/y if y odd, else x−y |
| `sq_sum` | x² + y² (mod p) |
| `sq_sum_xy` | x² + xy + y² (mod p) |
| `sq_sum_xy_x` | x² + xy + y² + x (mod p) |
| `cube_xy` | x³ + xy (mod p) |
| `cube_xy2_y` | x³ + xy² + y (mod p) |
| `s5_mul` | x · y in S₅ |
| `s5_conj` | x · y · x⁻¹ in S₅ |
| `s5_sandwich` | x · y · x in S₅ |

Default is `add`, p=97.

---

## Adding a new rule

Open `data/dataset.py`. Two steps:

**Step 1 — write the function**
```python
def op_my_rule(a, b, p):
    return (a * b + a) % p   # whatever your formula is
```

**Step 2 — register it** (in the `OPERATIONS` dict at the bottom of that section)
```python
OPERATIONS = {
    ...existing entries...
    "my_rule": (op_my_rule, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
}
```

The three things in the tuple:
1. Your function
2. `False` = integers mod p, `True` = S₅ permutations
3. The domain — which (a, b) pairs are valid inputs

Then run it:
```bash
python main.py --operation my_rule
```

---

## Where results go

Every run saves a JSON to `results/` named after its config:
```
results/add_p97_wd1.0_lr0.001_d128_l2_tf0.5_ep5000.json
```

Each file contains the full accuracy/loss curves plus the detected memorization epoch, grokking epoch, and gap between them.

---

## File map

```
main.py              ← start here for all experiments
data/dataset.py      ← add new operations here
models/transformer.py   ← model architecture (rarely need to touch)
experiments/train.py    ← training loop + TrainConfig (rarely need to touch)
results/             ← output JSONs (auto-created)
```
