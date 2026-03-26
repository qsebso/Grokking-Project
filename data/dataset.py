"""
Dataset generation for grokking experiments.

Supports multiple binary operations on integers mod p,
as well as operations on permutation groups (S5).
"""

import random
import itertools
import torch
from torch.utils.data import TensorDataset
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Operations on Z/pZ
# ─────────────────────────────────────────────────────────────────────────────

def op_add(a, b, p):
    """x ∘ y = x + y (mod p)"""
    return (a + b) % p

def op_sub(a, b, p):
    """x ∘ y = x - y (mod p)"""
    return (a - b) % p

def op_mul(a, b, p):
    """x ∘ y = x * y (mod p)"""
    return (a * b) % p

def op_div(a, b, p):
    """x ∘ y = x / y (mod p)  —  b must be nonzero (0 < b < p)"""
    return (a * pow(int(b), -1, p)) % p

def op_div_or_sub(a, b, p):
    """x ∘ y = x/y (mod p) if y is odd, else x - y (mod p)"""
    if b == 0:
        return (a - b) % p
    if b % 2 == 1:
        return (a * pow(int(b), -1, p)) % p
    else:
        return (a - b) % p

def op_add_or_sub(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x - y (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a - b) % p

def op_sub_or_add(a, b, p):
    """x ∘ y = x - y (mod p) if y is odd, else x + y (mod p)"""
    if b % 2 == 1:
        return (a - b) % p
    else:
        return (a + b) % p

def op_add_or_add2(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x + 2y (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a + 2*b) % p

def op_add_or_mul(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x * y (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a * b) % p

def op_add_or_mul_symmetric_on_a_plus_b_is_even(a, b, p):
    """x ∘ y = x * y (mod p) if a + b is even, else x + y (mod p)"""
    if (a + b) % 2 == 0:
        return (a * b) % p
    else:
        return (a + b) % p

def op_add_or_mul_symmetric_on_a_minus_b_is_even(a, b, p):
    """x ∘ y = x * y (mod p) if a - b is even, else x + y (mod p)"""
    if (a - b) % 2 == 0:
        return (a * b) % p
    else:
        return (a + b) % p

def op_3way_sub_add_mul(a, b, p):
    """Branch on region = (a + b) mod 3: 0 -> add, 1 -> sub, 2 -> mul (all mod p)."""
    region = (a + b) % 3
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a - b) % p
    return (a * b) % p


def op_3way_add_add_2_mul_mul(a, b, p):
    """Branch on region = (a + b) mod 3: 0 -> add, 1 -> a+2b, 2 -> ab (all mod p)."""
    region = (a + b) % 3
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a + (2 * b)) % p
    return (a * b) % p

def op_4way_sub_add_mul_mul2(a, b, p):
    """Branch on region = (a + b) mod 4: 0 -> add, 1 -> sub, 2 -> mul, 3 -> 2*(a + b) (all mod p)."""
    region = (a + b) % 4
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a - b) % p  
    if region == 2:
        return (a * b) % p
    if region == 3:
        return (2*(a + b)) % p

def op_4way_add_add2mul_sub2mul(a, b, p):
    """Branch on region = (a + b) mod 4: 0 -> add, 1 -> a+2b, 2 -> mul, 3 -> a - 2*b (all mod p)."""
    region = (a + b) % 4
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a + (2 * b)) % p  
    if region == 2:
        return (a * b) % p
    if region == 3:
        return (a - (2 * b)) % p

def op_4way_add_sub_mul_div(a, b, p):
    """Branch on region = (a + b) mod 4: 0 -> add, 1 -> sub, 2 -> mul, 3 -> div (all mod p)."""
    region = (a + b) % 4
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a - b) % p
    if region == 2:
        return (a * b) % p
    if region == 3:
        return (a * pow(int(b), -1, p)) % p

def op_4way_all_affine(a, b, p):
    """Branch on region = (a + b) mod 4: 0 -> a + b, 1 -> a - b, 2 -> a + 2b, 3 -> 2a + b (all mod p)."""
    region = (a + b) % 4
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a - b) % p
    if region == 2:
        return (a + 2 * b) % p
    return (2 * a + b) % p


# goal is to see if adding 1 is better than multiplying
def op_add_or_add_1(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x + y + 1 (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a + b + 1) % p

# goal is to see if 1 operation in a 2 rule set is better
def op_add_or_nothing(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return a

# goal is to make a 2 rule set that is seperated in 1 line in the dataset instead of odd/even
def op_add_or_mul_on_a_greater_than_b(a, b, p):
    """x ∘ y = x + y (mod p) if x >= y, else x * y (mod p)"""
    if a >= b:  
        return (a + b) % p
    else:
        return (a * b) % p

# goal is to see if adding 1 is better than multiplying
def op_add_or_add_1_on_a_greater_than_b(a, b, p):
    """x ∘ y = x + y (mod p) if x >= y, else x + y + 1 (mod p)"""
    if a >= b:
        return (a + b) % p
    else:
        return (a + 1) % p

def op_add_1(a, b, p):
    """x ∘ y = x + y + 1 (mod p)"""
    return (a + b + 1) % p

# goal is to see if 1 operation in a 2 rule set is better
def op_add_or_nothing_on_a_greater_than_b(a, b, p):
    """x ∘ y = x + y (mod p) if x >= y, else x"""
    if a >= b:
        return (a + b) % p
    else:
        return a

def op_add_or_a_plus_1(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x + 1 (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a + 1) % p

def op_add_or_add5(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else x + 5y (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a + 5*b) % p

def op_mul5(a, b, p):
    """x ∘ y = x * y * 5 (mod p)"""
    return (a * b * 5) % p

def op_mul5_plus_add(a, b, p):
    """x ∘ y = x * y * 5 + y (mod p)"""
    return (a + 5*b) % p

def op_add_or_affine(a, b, p):
    """x ∘ y = x + y (mod p) if y is odd, else 3x + 7y + 11 (mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (3*a + 7*b + 11) % p

def op_affine(a, b, p):
    """x ∘ y = 3x + 7y + 11 (mod p)"""
    return (3*a + 7*b + 11) % p

def op_sq_sum(a, b, p):
    """x ∘ y = x^2 + y^2 (mod p)"""
    return (a*a + b*b) % p

def op_sq_sum_xy(a, b, p):
    """x ∘ y = x^2 + xy + y^2 (mod p)"""
    return (a*a + a*b + b*b) % p

def op_sq_sum_xy_x(a, b, p):
    """x ∘ y = x^2 + xy + y^2 + x (mod p)"""
    return (a*a + a*b + b*b + a) % p

def op_cube_xy(a, b, p):
    """x ∘ y = x^3 + xy (mod p)"""
    return (a*a*a + a*b) % p

def op_cube_xy2_y(a, b, p):
    """x ∘ y = x^3 + xy^2 + y (mod p)"""
    return (a*a*a + a*b*b + b) % p


# ─────────────────────────────────────────────────────────────────────────────
# Permutation group S5
# ─────────────────────────────────────────────────────────────────────────────

def _compose(p, q):
    """Compose two permutations (as tuples): p ∘ q means apply q first then p."""
    return tuple(p[q[i]] for i in range(len(p)))

def _inverse(p):
    """Return the inverse of permutation p."""
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)

def _all_S5():
    return list(itertools.permutations(range(5)))

def op_s5_mul(a, b, p=None):
    """x ∘ y = x · y  in S5  (p unused, just for API compatibility)"""
    return _compose(a, b)

def op_s5_conj(a, b, p=None):
    """x ∘ y = x · y · x^{-1}  in S5"""
    return _compose(_compose(a, b), _inverse(a))

def op_s5_sandwich(a, b, p=None):
    """x ∘ y = x · y · x  in S5"""
    return _compose(_compose(a, b), a)


# ─────────────────────────────────────────────────────────────────────────────
# Registry of all named operations
# ─────────────────────────────────────────────────────────────────────────────

# Each entry:  name -> (op_fn, is_s5, domain_fn)
#   op_fn(a, b, p) -> result
#   is_s5          -> bool  (True = elements are permutations, not integers)
#   domain_fn(p)   -> list of valid (a, b) input pairs
OPERATIONS = {
    # ── integer mod-p ops ──────────────────────────────────────────────────
    "add":           (op_add,           False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "sub":           (op_sub,           False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "mul":           (op_mul,           False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "div":           (op_div,           False, lambda p: [(a, b) for a in range(p) for b in range(1, p)]),
    "div_or_sub":    (op_div_or_sub,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_sub":    (op_add_or_sub,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "sub_or_add":    (op_sub_or_add,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_add2":   (op_add_or_add2,   False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul":    (op_add_or_mul,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul_symmetric_on_a_plus_b_is_even": (op_add_or_mul_symmetric_on_a_plus_b_is_even, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul_symmetric_on_a_minus_b_is_even": (op_add_or_mul_symmetric_on_a_minus_b_is_even, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "3way_sub_add_mul": (op_3way_sub_add_mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "3way_add_add_2_mul_mul": (op_3way_add_add_2_mul_mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_sub_add_mul_mul2": (op_4way_sub_add_mul_mul2, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_add_add2mul_sub2mul": (op_4way_add_add2mul_sub2mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_add_sub_mul_div": (op_4way_add_sub_mul_div, False, lambda p: [(a, b) for a in range(p) for b in range(1, p)]),
    "4way_all_affine": (op_4way_all_affine, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_add_1":  (op_add_or_add_1,  False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_nothing": (op_add_or_nothing, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul_on_a_greater_than_b": (op_add_or_mul_on_a_greater_than_b, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_add_1_on_a_greater_than_b": (op_add_or_add_1_on_a_greater_than_b, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_1":           (op_add_1,           False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_nothing_on_a_greater_than_b": (op_add_or_nothing_on_a_greater_than_b, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_a_plus_1": (op_add_or_a_plus_1, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_add5": (op_add_or_add5, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_affine": (op_add_or_affine, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "affine":        (op_affine,        False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "mul5":          (op_mul5,          False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "mul5_plus_add": (op_mul5_plus_add, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "sq_sum":        (op_sq_sum,        False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "sq_sum_xy":     (op_sq_sum_xy,     False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "sq_sum_xy_x":   (op_sq_sum_xy_x,  False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "cube_xy":       (op_cube_xy,       False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "cube_xy2_y":    (op_cube_xy2_y,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    # ── S5 ops ─────────────────────────────────────────────────────────────
    "s5_mul":        (op_s5_mul,        True,  lambda p: [(a, b) for a in _all_S5() for b in _all_S5()]),
    "s5_conj":       (op_s5_conj,       True,  lambda p: [(a, b) for a in _all_S5() for b in _all_S5()]),
    "s5_sandwich":   (op_s5_sandwich,   True,  lambda p: [(a, b) for a in _all_S5() for b in _all_S5()]),
}


def resolve_branch_metric(operation: str, branch_metric_cfg: str) -> str:
    """
    How training should split per-sample accuracies (see experiments/train.py).

    branch_metric_cfg:
      "auto"     → infer from operation when known, else "b_parity"
      "b_parity" → odd vs even second operand b
      "a_ge_b"   → a >= b vs a < b
      "a_gt_b"   → a > b vs a <= b
    """
    if branch_metric_cfg != "auto":
        return branch_metric_cfg
    if operation == "add_or_mul_on_a_greater_than_b":
        return "a_ge_b"
    return "b_parity"


def branch_metric_labels(metric: str) -> Tuple[str, str]:
    """Short labels for the two branches (first list → train_odd_accs / val_odd_accs slot)."""
    if metric == "b_parity":
        return ("odd", "even")
    if metric == "a_ge_b":
        return ("a>=b", "a<b")
    if metric == "a_gt_b":
        return ("a>b", "a<=b")
    raise ValueError(
        f"Unknown branch metric '{metric}'. "
        "Use: auto, b_parity, a_ge_b, a_gt_b"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_vocab_integer(p):
    """
    Vocabulary for integer-domain ops.
    Tokens 0..p-1 → the numbers themselves.
    Token p   → '+' operator placeholder.
    Token p+1 → '=' token.
    Returns (op_token, eq_token, vocab_size).
    """
    return p, p + 1, p + 2


def _require_label_mod(label_mod: int) -> int:
    if label_mod < 2:
        raise ValueError("label_mod must be an integer >= 2")
    return label_mod


def category_label_num_classes(label_mode: str, label_mod: int = 3) -> int:
    """Number of classes for categorical targets (integer-domain experiments)."""
    if label_mode in ("c_parity", "b_parity", "a_parity"):
        return 2
    if label_mode in ("c_mod3", "a_plus_b_mod3"):
        return 3
    if label_mode == "c_mod":
        return _require_label_mod(label_mod)
    if label_mode == "a_plus_b_mod":
        return _require_label_mod(label_mod)
    raise ValueError(
        f"Unknown label_mode '{label_mode}'. "
        "Use: c_parity, b_parity, a_parity, c_mod3, a_plus_b_mod3, "
        "c_mod, a_plus_b_mod (last two need --label_mod k)"
    )


def compute_category_label(
    label_mode: str,
    a: int,
    b: int,
    c: int,
    p: int,
    label_mod: int = 3,
) -> int:
    """
    Map (a, b, c) to a class index.

    Modes ``c_mod`` / ``a_plus_b_mod`` use ``label_mod = k`` for **k** classes (e.g. k=11 -> hard
    input statistic). For ``c_*``, the model must track true ``c``; for ``a_plus_b_*``, only inputs.

    ``c_mod3`` / ``a_plus_b_mod3`` are fixed-3 aliases (same as ``label_mod=3``).
    """
    del p
    if label_mode == "c_parity":
        return c % 2
    if label_mode == "b_parity":
        return b % 2
    if label_mode == "a_parity":
        return a % 2
    if label_mode == "c_mod3":
        return c % 3
    if label_mode == "a_plus_b_mod3":
        return (a + b) % 3
    if label_mode == "c_mod":
        k = _require_label_mod(label_mod)
        return c % k
    if label_mode == "a_plus_b_mod":
        k = _require_label_mod(label_mod)
        return (a + b) % k
    raise ValueError(
        f"Unknown label_mode '{label_mode}'. "
        "Use: c_parity, b_parity, a_parity, c_mod3, a_plus_b_mod3, c_mod, a_plus_b_mod"
    )


def _encode_integer_pairs(
    pairs,
    op_fn,
    p,
    fmt: str = "a_op_b_eq",
    label_mode: Optional[str] = None,
    label_mod: int = 3,
):
    """
    Encode integer pairs into (input_seq, label) tensors.

    fmt controls the token sequence layout:
      "a_op_b_eq"  →  [a, op, b, =]          (default, matches the paper)
      "a_b_eq"     →  [a, b, =]               (no explicit operator token)
      "a_op_b_eq_rule" → [rule, a, op, b, =]  (prepend a rule-type token)

    For "a_op_b_eq_rule" the rule token is always vocab_size-1 (one extra
    token appended to the end of the vocabulary).
    """
    op_tok, eq_tok, base_vocab = _build_vocab_integer(p)

    if fmt == "a_op_b_eq":
        # Standard: [a, op, b, =]  — vocab unchanged
        vocab_extra = 0
        def make_seq(a, b):
            return [a, op_tok, b, eq_tok]

    elif fmt == "a_b_eq":
        # No operator token: [a, b, =]
        # We still keep the same vocab size so that label indices match.
        vocab_extra = 0
        def make_seq(a, b):
            return [a, b, eq_tok]

    elif fmt == "a_op_b_eq_rule":
        # Prepend a rule-type token: [rule, a, op, b, =]
        # The rule token ID lives just past the normal vocab.
        vocab_extra = 1
        rule_tok = base_vocab   # = p + 2  (right after eq_tok = p+1)
        def make_seq(a, b):
            return [rule_tok, a, op_tok, b, eq_tok]

    else:
        raise ValueError(
            f"Unknown input_format '{fmt}'. "
            "Choose from: 'a_op_b_eq', 'a_b_eq', 'a_op_b_eq_rule'"
        )

    xs, ys = [], []
    for a, b in pairs:
        c = op_fn(a, b, p)
        xs.append(make_seq(a, b))
        if label_mode is None:
            ys.append(c)
        else:
            ys.append(compute_category_label(label_mode, a, b, c, p, label_mod))
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            vocab_extra)


def _encode_s5_pairs(pairs, op_fn, fmt: str = "a_op_b_eq"):
    """
    Encode S5 pairs.  Each permutation (tuple of 5 ints) is mapped to a
    single integer index 0..119.  Vocab: 120 elements + op token + eq token.

    fmt behaves the same as for integer pairs (rule token supported too).
    """
    all_perms = _all_S5()
    perm_to_idx = {perm: i for i, perm in enumerate(all_perms)}
    base_vocab = len(all_perms)   # 120
    op_tok = base_vocab
    eq_tok = base_vocab + 1

    if fmt == "a_op_b_eq":
        vocab_extra = 0
        def make_seq(ia, ib):
            return [ia, op_tok, ib, eq_tok]

    elif fmt == "a_b_eq":
        vocab_extra = 0
        def make_seq(ia, ib):
            return [ia, ib, eq_tok]

    elif fmt == "a_op_b_eq_rule":
        vocab_extra = 1
        rule_tok = base_vocab + 2   # right after eq_tok
        def make_seq(ia, ib):
            return [rule_tok, ia, op_tok, ib, eq_tok]

    else:
        raise ValueError(
            f"Unknown input_format '{fmt}'. "
            "Choose from: 'a_op_b_eq', 'a_b_eq', 'a_op_b_eq_rule'"
        )

    xs, ys = [], []
    for a, b in pairs:
        c = op_fn(a, b)
        ia, ib, ic = perm_to_idx[a], perm_to_idx[b], perm_to_idx[c]
        xs.append(make_seq(ia, ib))
        ys.append(ic)
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            vocab_extra)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(
    operation: str = "add",
    p: int = 97,
    train_frac: float = 0.5,
    max_train_samples: int = None,
    input_format: str = "a_op_b_eq",
    seed: int = 42,
    label_noise: float = 0.0,
):
    """
    Build train/val TensorDatasets for a given binary operation.

    Parameters
    ----------
    operation          : key in OPERATIONS (e.g. "add", "sub", "s5_mul", …)
    p                  : prime / modulus (ignored for S5 ops)
    train_frac         : fraction of all pairs used for training
    max_train_samples  : hard cap on training set size (applied *after*
                         train_frac split, so you can use both together).
                         None = no cap.
    input_format       : how inputs are tokenised —
                           "a_op_b_eq"       → [a, op, b, =]   (paper default)
                           "a_b_eq"          → [a, b, =]        (no op token)
                           "a_op_b_eq_rule"  → [rule, a, op, b, =]
    seed               : random seed for the train/val split
    label_noise        : fraction of *training* labels to randomly corrupt

    Returns
    -------
    train_ds, val_ds   : TensorDataset objects
    vocab_size         : size of the token vocabulary
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Valid options: {sorted(OPERATIONS.keys())}"
        )

    op_fn, is_s5, domain_fn = OPERATIONS[operation]

    # ── build all pairs ────────────────────────────────────────────────────
    all_pairs = domain_fn(p)

    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:]

    # ── hard cap on training set size ─────────────────────────────────────
    if max_train_samples is not None and max_train_samples < len(train_pairs):
        train_pairs = train_pairs[:max_train_samples]

    # ── encode ─────────────────────────────────────────────────────────────
    if is_s5:
        x_train, y_train, ve = _encode_s5_pairs(train_pairs, op_fn, input_format)
        x_val,   y_val,   _  = _encode_s5_pairs(val_pairs,   op_fn, input_format)
        vocab_size = len(_all_S5()) + 2 + ve
    else:
        x_train, y_train, ve = _encode_integer_pairs(train_pairs, op_fn, p, input_format)
        x_val,   y_val,   _  = _encode_integer_pairs(val_pairs,   op_fn, p, input_format)
        _, _, base_vocab = _build_vocab_integer(p)
        vocab_size = base_vocab + ve

    # ── optional label noise on training set ───────────────────────────────
    if label_noise > 0.0:
        n_noisy = int(len(y_train) * label_noise)
        noisy_idx = torch.randperm(len(y_train))[:n_noisy]
        # corrupt only among valid answer tokens (0 .. p-1 or 0 .. 119)
        n_answers = vocab_size - 2 - ve
        y_train[noisy_idx] = torch.randint(0, n_answers, (n_noisy,))

    return (TensorDataset(x_train, y_train),
            TensorDataset(x_val,   y_val),
            vocab_size)


def make_category_dataset(
    operation: str = "add",
    p: int = 97,
    train_frac: float = 0.5,
    max_train_samples: int = None,
    input_format: str = "a_op_b_eq",
    seed: int = 42,
    label_noise: float = 0.0,
    label_mode: str = "c_parity",
    label_mod: int = 3,
):
    """
    Same token sequences as ``make_dataset``, but targets are categorical class indices.

    ``label_mode`` (recommended first try: ``c_parity`` or ``c_mod`` with small ``label_mod``):
      - **Output-based:** ``c_parity``, ``c_mod3``, ``c_mod`` (use ``label_mod=k`` for ``c % k``).
      - **Input-based:** ``b_parity``, ``a_parity``, ``a_plus_b_mod3``, ``a_plus_b_mod``
        (``a_plus_b_mod`` with large ``label_mod`` gives a high-cardinality input statistic).

    Returns ``train_ds, val_ds, vocab_size, num_classes`` (``num_classes`` is the softmax size).
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Valid options: {sorted(OPERATIONS.keys())}"
        )

    op_fn, is_s5, domain_fn = OPERATIONS[operation]
    if is_s5:
        raise ValueError("make_category_dataset supports integer mod-p ops only (not S5).")

    num_classes = category_label_num_classes(label_mode, label_mod)

    all_pairs = domain_fn(p)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    if max_train_samples is not None and max_train_samples < len(train_pairs):
        train_pairs = train_pairs[:max_train_samples]

    x_train, y_train, ve = _encode_integer_pairs(
        train_pairs, op_fn, p, input_format, label_mode=label_mode, label_mod=label_mod
    )
    x_val, y_val, _ = _encode_integer_pairs(
        val_pairs, op_fn, p, input_format, label_mode=label_mode, label_mod=label_mod
    )
    _, _, base_vocab = _build_vocab_integer(p)
    vocab_size = base_vocab + ve

    if label_noise > 0.0:
        n_noisy = int(len(y_train) * label_noise)
        noisy_idx = torch.randperm(len(y_train))[:n_noisy]
        y_train[noisy_idx] = torch.randint(0, num_classes, (n_noisy,))

    return (
        TensorDataset(x_train, y_train),
        TensorDataset(x_val, y_val),
        vocab_size,
        num_classes,
    )
