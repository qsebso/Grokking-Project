"""
Dataset generation for grokking experiments.

Supports multiple binary operations on integers mod p,
as well as operations on permutation groups (S5).
"""

import random
import itertools
import torch
from torch.utils.data import TensorDataset
from typing import List, Optional, Sequence, Tuple


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

def op_add_or_add(a, b, p):  # for continous to test if seperating groups in rules acts the same as continous if rules are the same
    """x ∘ y = x + y (mod p) if y is odd, else x + y(mod p)"""
    if b % 2 == 1:
        return (a + b) % p
    else:
        return (a + b) % p

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

def op_3way_add_mul_div(a, b, p):
    """Branch on region = (a + b) mod 3: 0 -> add, 1 -> mul, 2 -> div (all mod p).

    When region is div and b ≡ 0 (mod p), inverse is undefined; use (a - b) % p == a % p
    (same convention as ``op_div_or_sub`` for b == 0).
    """
    region = (a + b) % 3
    if region == 0:
        return (a + b) % p
    if region == 1:
        return (a * b) % p
    if b % p == 0:
        return (a - b) % p
    return (a * pow(int(b), -1, p)) % p


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
        if b % p == 0:
            return (a - b) % p
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


def op_10way_mixed_hard(a, b, p):
    """
    Ten-way split: ``region = (a + b) % 10``. Full ``p×p`` domain (no division).

    Rough difficulty ramp (all mod p):
      0: a + b
      1: a - b
      2: a * b
      3: a + 2b
      4: a² + b
      5: a + b²
      6: ab + a + b    [ (a+1)(b+1) - 1 ]
      7: 2a + 3b + 5
      8: a³ + b²
      9: ab² + 2a

    For categorical probes, ``label_mode=a_plus_b_mod`` with ``label_mod=10`` matches this branch key.
    """
    r = (a + b) % 10
    if r == 0:
        return (a + b) % p
    if r == 1:
        return (a - b) % p
    if r == 2:
        return (a * b) % p
    if r == 3:
        return (a + 2 * b) % p
    if r == 4:
        return (a * a + b) % p
    if r == 5:
        return (a + b * b) % p
    if r == 6:
        return (a * b + a + b) % p
    if r == 7:
        return (2 * a + 3 * b + 5) % p
    if r == 8:
        return (a * a * a + b * b) % p
    return (a * b * b + 2 * a) % p


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
    "add_or_add":    (op_add_or_add,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_add2":   (op_add_or_add2,   False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul":    (op_add_or_mul,    False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul_symmetric_on_a_plus_b_is_even": (op_add_or_mul_symmetric_on_a_plus_b_is_even, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "add_or_mul_symmetric_on_a_minus_b_is_even": (op_add_or_mul_symmetric_on_a_minus_b_is_even, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "3way_sub_add_mul": (op_3way_sub_add_mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "3way_add_add_2_mul_mul": (op_3way_add_add_2_mul_mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "3way_add_mul_div": (op_3way_add_mul_div, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_sub_add_mul_mul2": (op_4way_sub_add_mul_mul2, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_add_add2mul_sub2mul": (op_4way_add_add2mul_sub2mul, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "4way_add_sub_mul_div": (op_4way_add_sub_mul_div, False, lambda p: [(a, b) for a in range(p) for b in range(1, p)]),
    "4way_all_affine": (op_4way_all_affine, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
    "10way_mixed_hard": (op_10way_mixed_hard, False, lambda p: [(a, b) for a in range(p) for b in range(p)]),
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

# ── Multi-rule ops: (num_branches, rule_id(a,b,p)) for disjoint output ranges ─────────
# rule_id in 0..num_branches-1; encoded label = rule_id * p + (c mod p) when rule_count > 1.


def _rid_b_odd_branch0(a: int, b: int, p: int) -> int:
    del a, p
    return 1 - (b & 1)


def _rid_a_ge_b_branch0(a: int, b: int, p: int) -> int:
    del p
    return 0 if a >= b else 1


def _rid_a_gt_b_branch0(a: int, b: int, p: int) -> int:
    del p
    return 0 if a > b else 1


def _rid_sum_mod_m(m: int):
    def inner(a: int, b: int, p: int) -> int:
        del p
        return (a + b) % m

    return inner


def _rid_sum_parity_mul_on_even(a: int, b: int, p: int) -> int:
    del p
    return (a + b) % 2


def _rid_diff_parity_mul_on_even(a: int, b: int, p: int) -> int:
    del p
    return (a - b) % 2


def _rid_div_or_sub(a: int, b: int, p: int) -> int:
    del a, p
    if b == 0:
        return 0
    if b % 2 == 1:
        return 1
    return 2


OPERATION_RULE_INFO = {
    "add_or_mul": (2, _rid_b_odd_branch0),
    "add_or_sub": (2, _rid_b_odd_branch0),
    "sub_or_add": (2, _rid_b_odd_branch0),
    # Same b-parity split as add_or_add2, but both branches return (a+b)%p —
    # disjoint bands still separate identical outputs by branch for ablation.
    "add_or_add": (2, _rid_b_odd_branch0),
    "add_or_add2": (2, _rid_b_odd_branch0),
    "div_or_sub": (3, _rid_div_or_sub),
    "add_or_mul_symmetric_on_a_plus_b_is_even": (2, _rid_sum_parity_mul_on_even),
    "add_or_mul_symmetric_on_a_minus_b_is_even": (2, _rid_diff_parity_mul_on_even),
    "add_or_mul_on_a_greater_than_b": (2, _rid_a_ge_b_branch0),
    "add_or_add_1_on_a_greater_than_b": (2, _rid_a_ge_b_branch0),
    "add_or_nothing_on_a_greater_than_b": (2, _rid_a_ge_b_branch0),
    "add_or_add_1": (2, _rid_b_odd_branch0),
    "add_or_nothing": (2, _rid_b_odd_branch0),
    "add_or_a_plus_1": (2, _rid_b_odd_branch0),
    "add_or_add5": (2, _rid_b_odd_branch0),
    "add_or_affine": (2, _rid_b_odd_branch0),
    "3way_sub_add_mul": (3, _rid_sum_mod_m(3)),
    "3way_add_mul_div": (3, _rid_sum_mod_m(3)),
    "3way_add_add_2_mul_mul": (3, _rid_sum_mod_m(3)),
    "4way_sub_add_mul_mul2": (4, _rid_sum_mod_m(4)),
    "4way_add_add2mul_sub2mul": (4, _rid_sum_mod_m(4)),
    "4way_add_sub_mul_div": (4, _rid_sum_mod_m(4)),
    "4way_all_affine": (4, _rid_sum_mod_m(4)),
    "10way_mixed_hard": (10, _rid_sum_mod_m(10)),
}


def operation_num_rules(operation: str) -> int:
    """How many disjoint rules the operation has (1 if unknown / single-rule)."""
    if operation in OPERATION_RULE_INFO:
        return OPERATION_RULE_INFO[operation][0]
    return 1


def resolve_rule_id(operation: str, a: int, b: int, p: int) -> int:
    if operation not in OPERATION_RULE_INFO:
        return 0
    n, fn = OPERATION_RULE_INFO[operation]
    rid = int(fn(a, b, p))
    if not 0 <= rid < n:
        raise RuntimeError(f"rule_id {rid} out of range for {operation!r} (n={n})")
    return rid


def validate_rule_count(operation: str, rule_count: int) -> None:
    """
    rule_count == 1: standard overlapping outputs in 0..p-1.
    rule_count == n: disjoint bands [0,p-1], [p,2p-1], … when n matches the op's branch count.
    """
    if rule_count < 1:
        raise ValueError("rule_count must be >= 1")
    n = operation_num_rules(operation)
    if rule_count == 1:
        return
    if rule_count != n:
        raise ValueError(
            f"rule_count={rule_count} incompatible with operation {operation!r} "
            f"(that op has {n} branch(es); use rule_count={n} for disjoint bands, or 1 for default)."
        )


def encode_disjoint_rule_output(local_c: int, rule_id: int, p: int) -> int:
    """Map branch-local result in 0..p-1 to a global class in 0 .. rule_count*p - 1."""
    c = int(local_c) % p
    return int(rule_id) * p + c


# ── Train label corruption (asymmetric + symmetric) ─────────────────────────

NOISE_MODE_CHOICES = (
    "random_wrong_c",
    "fixed_wrong_c",
    "shifted_wrong_c",
    "other_rule_c",
    "fixed_wrong_c_cross_rule",
)

# add/mul swap for ``other_rule_c`` (branching must match each op's implementation)
OTHER_RULE_SUPPORTED_OPS = frozenset(
    {
        "add_or_mul",
        "add_or_mul_symmetric_on_a_plus_b_is_even",
        "add_or_mul_symmetric_on_a_minus_b_is_even",
        "add_or_mul_on_a_greater_than_b",
    }
)


def _branch_uses_mul(operation: str, a: int, b: int, p: int) -> bool:
    """Whether the active branch applies multiplication (vs addition) for supported ops."""
    del p
    if operation == "add_or_mul":
        return b % 2 == 0
    if operation == "add_or_mul_symmetric_on_a_plus_b_is_even":
        return (a + b) % 2 == 0
    if operation == "add_or_mul_symmetric_on_a_minus_b_is_even":
        return (a - b) % 2 == 0
    if operation == "add_or_mul_on_a_greater_than_b":
        return a < b
    raise ValueError(
        f"other_rule_c not implemented for operation={operation!r} "
        f"(supported: {sorted(OTHER_RULE_SUPPORTED_OPS)})"
    )


def _other_rule_local_c(operation: str, a: int, b: int, p: int) -> int:
    """Branch-local result under the *other* binary (add vs mul) for the same (a, b)."""
    c_add = (a + b) % p
    c_mul = (a * b) % p
    if _branch_uses_mul(operation, a, b, p):
        return c_add
    return c_mul


def _pick_fixed_wrong(true_y: int, primary: int, backup: int, num_classes: int) -> int:
    if primary != true_y:
        return primary
    if backup != true_y:
        return backup
    for c in range(num_classes):
        if c != true_y:
            return c
    raise ValueError("num_classes < 2: no incorrect label exists")


def validate_noise_mode_config(
    noise_mode: str,
    label_noise: float,
    *,
    operation: str,
    rule_count: int,
    is_s5: bool = False,
    label_mode: Optional[str] = None,
) -> None:
    if label_noise <= 0:
        return
    if noise_mode not in NOISE_MODE_CHOICES:
        raise ValueError(f"Unknown noise_mode {noise_mode!r}; choose from {NOISE_MODE_CHOICES}")
    if is_s5:
        if noise_mode != "random_wrong_c":
            raise ValueError(
                f"S5 only supports noise_mode='random_wrong_c' (got {noise_mode!r})."
            )
        return
    if noise_mode == "other_rule_c":
        if rule_count != 2:
            raise ValueError("other_rule_c requires rule_count=2.")
        if operation not in OTHER_RULE_SUPPORTED_OPS:
            raise ValueError(
                f"other_rule_c is only implemented for {sorted(OTHER_RULE_SUPPORTED_OPS)}; "
                f"got operation={operation!r}."
            )
        if label_mode is not None and label_mode != "c":
            raise ValueError("other_rule_c with categorical training requires label_mode=c.")
    elif noise_mode == "fixed_wrong_c_cross_rule":
        if rule_count < 2:
            raise ValueError("fixed_wrong_c_cross_rule requires rule_count>=2.")
    elif noise_mode == "fixed_wrong_c":
        if rule_count != 1:
            raise ValueError(
                "fixed_wrong_c is for rule_count=1; use fixed_wrong_c_cross_rule for multi-rule."
            )
    elif noise_mode == "shifted_wrong_c":
        if rule_count != 1:
            raise ValueError(
                "shifted_wrong_c is for rule_count=1 (labels in a single band 0..p-1)."
            )


def resolve_parsed_noise_mode(args, parser=None) -> str:
    """
    Resolve CLI noise mode: at most one of the ``--fixed_wrong_c``-style flags, else ``--noise_mode``.
    """
    aliases: List[str] = []
    for name in NOISE_MODE_CHOICES:
        if getattr(args, f"noise_alias_{name}", False):
            aliases.append(name)
    if len(aliases) > 1:
        msg = f"Use at most one noise-mode shorthand flag; got: {aliases}"
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    if aliases:
        if getattr(args, "noise_mode", None) not in (None, aliases[0]):
            msg = f"Conflicting noise mode: flags imply {aliases[0]!r} but --noise_mode={args.noise_mode!r}"
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        return aliases[0]
    mode = getattr(args, "noise_mode", None) or "random_wrong_c"
    return mode


def register_noise_mode_cli_args(parser) -> None:
    """Add ``--noise_mode``, shorthand flags, and fixed-target options (shared by main / category / factor)."""
    parser.add_argument(
        "--noise_mode",
        type=str,
        default=None,
        choices=list(NOISE_MODE_CHOICES),
        help=(
            "Asymmetric corruption when --noise>0 (default: random_wrong_c). "
            "Shorthand: --fixed_wrong_c, --shifted_wrong_c, --other_rule_c, …"
        ),
    )
    for name in NOISE_MODE_CHOICES:
        flag = f"--{name}"
        parser.add_argument(
            flag,
            dest=f"noise_alias_{name}",
            action="store_true",
            help=f"Set noise_mode={name} (mutually exclusive with other mode flags).",
        )
    parser.add_argument(
        "--noise_fixed_target",
        type=int,
        default=5,
        help="Primary wrong class for fixed_wrong_c / fixed_wrong_c_cross_rule (mod num_classes).",
    )
    parser.add_argument(
        "--noise_fixed_backup",
        type=int,
        default=None,
        help="Second wrong class if true label equals noise_fixed_target (default: next class).",
    )


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


def category_label_num_classes(
    label_mode: str,
    label_mod: int = 3,
    *,
    p: int,
    rule_count: int = 1,
) -> int:
    """Number of classes for categorical targets (integer-domain experiments)."""
    if label_mode == "c":
        return p * rule_count
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
        "Use: c, c_parity, b_parity, a_parity, c_mod3, a_plus_b_mod3, "
        "c_mod, a_plus_b_mod (c_mod / a_plus_b_mod need --label_mod k)"
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

    ``c`` uses the full (possibly disjoint) output as class id: ``0 .. p-1`` or
    ``0 .. rule_count*p-1`` when ``rule_count > 1`` in the dataloader.
    """
    if label_mode == "c":
        return c
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
        "Use: c, c_parity, b_parity, a_parity, c_mod3, a_plus_b_mod3, c_mod, a_plus_b_mod"
    )


def _noise_generator(seed: int) -> torch.Generator:
    """Isolated RNG for label noise; keeps noise reproducible for a given dataset ``seed``."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def apply_train_label_corruption(
    y_train: torch.Tensor,
    train_pairs: Optional[Sequence[Tuple[int, int]]],
    *,
    operation: str,
    p: int,
    rule_count: int,
    num_classes: int,
    label_noise: float,
    label_noise_sym: float,
    noise_mode: str,
    noise_fixed_target: int,
    noise_fixed_backup: Optional[int],
    generator: torch.Generator,
) -> None:
    """
    Mutate training labels in place (asymmetric corruption then optional symmetric swaps).

    ``noise_mode`` (with ``label_noise > 0``): which wrong-label rule to apply on a random
    subset of training rows (same count as before: ``floor(n * label_noise)``).

    Structured modes ``other_rule_c`` / ``shifted_wrong_c`` / … require ``train_pairs`` aligned
    with ``y_train`` (same length, row i → (a,b)); pass ``None`` only for ``random_wrong_c`` /
    ``fixed_*`` / ``shifted_wrong_c`` when pairs are unused (``shifted`` and ``fixed`` do not
    need pairs; ``other_rule_c`` does).
    """
    if label_noise < 0 or label_noise_sym < 0:
        raise ValueError("label noise rates must be >= 0")
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")

    if label_noise > 0.0:
        n_noisy = int(len(y_train) * label_noise)
        if n_noisy > 0:
            noisy_idx = torch.randperm(len(y_train), generator=generator)[:n_noisy]

            if noise_mode == "random_wrong_c":
                y_flat = y_train.view(-1)
                for j in range(n_noisy):
                    ii = int(noisy_idx[j].item())
                    ty = int(y_flat[ii].item())
                    nw = num_classes - 1
                    if nw < 1:
                        continue
                    r = int(torch.randint(0, nw, (1,), generator=generator).item())
                    k = 0
                    for c in range(num_classes):
                        if c == ty:
                            continue
                        if k == r:
                            y_flat[ii] = c
                            break
                        k += 1

            elif noise_mode == "fixed_wrong_c":
                primary = int(noise_fixed_target) % num_classes
                backup = noise_fixed_backup
                if backup is None:
                    backup = (primary + 1) % num_classes
                else:
                    backup = int(backup) % num_classes
                if backup == primary:
                    backup = (primary + 1) % num_classes
                y_flat = y_train.view(-1)
                for j in range(n_noisy):
                    ii = int(noisy_idx[j].item())
                    ty = int(y_flat[ii].item())
                    y_flat[ii] = _pick_fixed_wrong(ty, primary, backup, num_classes)

            elif noise_mode == "fixed_wrong_c_cross_rule":
                primary = int(noise_fixed_target) % num_classes
                backup = noise_fixed_backup
                if backup is None:
                    backup = (primary + 1) % num_classes
                else:
                    backup = int(backup) % num_classes
                if backup == primary:
                    backup = (primary + 1) % num_classes
                y_flat = y_train.view(-1)
                for j in range(n_noisy):
                    ii = int(noisy_idx[j].item())
                    ty = int(y_flat[ii].item())
                    y_flat[ii] = _pick_fixed_wrong(ty, primary, backup, num_classes)

            elif noise_mode == "shifted_wrong_c":
                y_flat = y_train.view(-1)
                for j in range(n_noisy):
                    ii = int(noisy_idx[j].item())
                    ty = int(y_flat[ii].item())
                    shift = 1
                    wrong = (ty + shift) % num_classes
                    while wrong == ty and num_classes > 1:
                        shift += 1
                        wrong = (ty + shift) % num_classes
                    y_flat[ii] = wrong

            elif noise_mode == "other_rule_c":
                if train_pairs is None or len(train_pairs) != len(y_train):
                    raise ValueError(
                        "other_rule_c requires train_pairs aligned with y_train (same length)."
                    )
                y_flat = y_train.view(-1)
                for j in range(n_noisy):
                    ii = int(noisy_idx[j].item())
                    a, b = train_pairs[ii]
                    rid = resolve_rule_id(operation, a, b, p)
                    ol = _other_rule_local_c(operation, a, b, p)
                    y_flat[ii] = encode_disjoint_rule_output(ol, rid, p)

            else:
                raise ValueError(f"Unknown noise_mode {noise_mode!r}")

    if label_noise_sym > 0.0:
        n_noisy = int(len(y_train) * label_noise_sym)
        n_pairs = n_noisy // 2
        if n_pairs > 0:
            idx = torch.randperm(len(y_train), generator=generator)[: n_pairs * 2]
            idx_a, idx_b = idx[:n_pairs], idx[n_pairs:]
            ya = y_train[idx_a].clone()
            yb = y_train[idx_b].clone()
            y_train[idx_a] = yb
            y_train[idx_b] = ya


def _encode_integer_pairs(
    pairs,
    op_fn,
    p,
    fmt: str = "a_op_b_eq",
    label_mode: Optional[str] = None,
    label_mod: int = 3,
    operation: str = "add",
    rule_count: int = 1,
):
    """
    Encode integer pairs into (input_seq, label) tensors.

    fmt controls the token sequence layout:
      "a_op_b_eq"  →  [a, op, b, =]          (default, matches the paper)
      "a_b_eq"     →  [a, b, =]               (no explicit operator token)
      "a_op_b_eq_rule" → [rule, a, op, b, =]  (prepend a rule-type token)
      "a_op_b_eq_bparity" → [a, op, b, parity, =]  (parity = b mod 2 as token 0 or 1)
      "a_op_bparity_eq"   → [a, op, parity, =]     (parity only; drops full b)

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

    elif fmt == "a_op_b_eq_bparity":
        # [a, op, b, b%2, =] — last token still "=" for last-position readout
        vocab_extra = 0
        def make_seq(a, b):
            return [a, op_tok, b, b % 2, eq_tok]

    elif fmt == "a_op_bparity_eq":
        # [a, op, b%2, =] — no full b in the input
        vocab_extra = 0
        def make_seq(a, b):
            return [a, op_tok, b % 2, eq_tok]

    else:
        raise ValueError(
            f"Unknown input_format '{fmt}'. "
            "Choose from: 'a_op_b_eq', 'a_b_eq', 'a_op_b_eq_rule', "
            "'a_op_b_eq_bparity', 'a_op_bparity_eq'"
        )

    xs, ys = [], []
    for a, b in pairs:
        c_local = op_fn(a, b, p)
        c = (
            encode_disjoint_rule_output(c_local, resolve_rule_id(operation, a, b, p), p)
            if rule_count > 1
            else c_local
        )
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

    elif fmt in ("a_op_b_eq_bparity", "a_op_bparity_eq"):
        raise ValueError(
            f"input_format {fmt!r} is not supported for S5 operations "
            "(parity tokens would collide with permutation indices)."
        )

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
    label_noise_sym: float = 0.0,
    rule_count: int = 1,
    noise_mode: str = "random_wrong_c",
    noise_fixed_target: int = 5,
    noise_fixed_backup: Optional[int] = None,
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
                           "a_op_b_eq_bparity" → [a, op, b, b mod 2, =]
                           "a_op_bparity_eq"   → [a, op, b mod 2, =]
    seed               : random seed for the train/val split and for label-noise RNG
                         (same seed → same shuffle/split and same noisy train labels)
    label_noise        : fraction of *training* rows to corrupt (asymmetric)
    label_noise_sym    : fraction of *training* labels for symmetric pair-swap noise
    noise_mode         : how corrupted labels are chosen (see ``NOISE_MODE_CHOICES``)
    noise_fixed_target : primary wrong class for ``fixed_wrong_c*`` (mod num_classes)
    noise_fixed_backup : alternate wrong class when true label equals target (default: next class)
    rule_count         : 1 = outputs in ``0..p-1`` (default).  ``n > 1`` must match the
                         operation's branch count; then labels are disjoint bands
                         ``[0,p-1], [p,2p-1], …`` so the active rule is identifiable.

    Returns
    -------
    train_ds, val_ds   : TensorDataset objects
    vocab_size         : size of the token vocabulary (for embeddings / input tokens)
    num_logits         : softmax width: ``rule_count * p`` if ``rule_count > 1``, else ``None``
                         (caller should use ``vocab_size`` when ``None``).
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Valid options: {sorted(OPERATIONS.keys())}"
        )

    op_fn, is_s5, domain_fn = OPERATIONS[operation]
    validate_rule_count(operation, rule_count)
    validate_noise_mode_config(
        noise_mode,
        label_noise,
        operation=operation,
        rule_count=rule_count,
        is_s5=is_s5,
        label_mode=None,
    )

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
        x_train, y_train, ve = _encode_integer_pairs(
            train_pairs, op_fn, p, input_format,
            operation=operation, rule_count=rule_count,
        )
        x_val,   y_val,   _  = _encode_integer_pairs(
            val_pairs,   op_fn, p, input_format,
            operation=operation, rule_count=rule_count,
        )
        _, _, base_vocab = _build_vocab_integer(p)
        vocab_size = base_vocab + ve

    num_logits = (rule_count * p) if (not is_s5 and rule_count > 1) else None

    # ── optional label noise on training set ───────────────────────────────
    noise_gen = _noise_generator(seed)
    if not is_s5:
        n_answers = rule_count * p if rule_count > 1 else (vocab_size - 2 - ve)
        apply_train_label_corruption(
            y_train,
            train_pairs,
            operation=operation,
            p=p,
            rule_count=rule_count,
            num_classes=n_answers,
            label_noise=label_noise,
            label_noise_sym=label_noise_sym,
            noise_mode=noise_mode,
            noise_fixed_target=noise_fixed_target,
            noise_fixed_backup=noise_fixed_backup,
            generator=noise_gen,
        )
    else:
        if label_noise_sym > 0.0:
            raise ValueError("label_noise_sym is not supported for S5 operations.")
        n_answers = vocab_size - 2 - ve
        apply_train_label_corruption(
            y_train,
            None,
            operation=operation,
            p=p,
            rule_count=1,
            num_classes=n_answers,
            label_noise=label_noise,
            label_noise_sym=0.0,
            noise_mode="random_wrong_c",
            noise_fixed_target=noise_fixed_target,
            noise_fixed_backup=noise_fixed_backup,
            generator=noise_gen,
        )

    return (TensorDataset(x_train, y_train),
            TensorDataset(x_val,   y_val),
            vocab_size,
            num_logits)


def make_category_dataset(
    operation: str = "add",
    p: int = 97,
    train_frac: float = 0.5,
    max_train_samples: int = None,
    input_format: str = "a_op_b_eq",
    seed: int = 42,
    label_noise: float = 0.0,
    label_noise_sym: float = 0.0,
    label_mode: str = "c_parity",
    label_mod: int = 3,
    rule_count: int = 1,
    noise_mode: str = "random_wrong_c",
    noise_fixed_target: int = 5,
    noise_fixed_backup: Optional[int] = None,
):
    """
    Same token sequences as ``make_dataset``, but targets are categorical class indices.

    ``label_mode`` (recommended first try: ``c_parity`` or ``c_mod`` with small ``label_mod``):
      - **Output-based:** ``c`` (full ``c`` in ``0..p-1``, ``num_classes=p``), ``c_parity``,
        ``c_mod3``, ``c_mod`` (use ``label_mod=k`` for ``c % k``).
      - **Input-based:** ``b_parity``, ``a_parity``, ``a_plus_b_mod3``, ``a_plus_b_mod``
        (``a_plus_b_mod`` with large ``label_mod`` gives a high-cardinality input statistic).

    ``rule_count`` (same semantics as ``make_dataset``): when ``> 1``, encodes outputs in
    disjoint bands before applying ``label_mode`` on that scalar (e.g. ``label_mode=c`` →
    ``num_classes = rule_count * p``).

    ``label_noise`` / ``label_noise_sym`` apply only to the **training** split (see
    ``apply_train_label_corruption``). The same ``seed`` controls the pair shuffle/split and
    the label-noise pattern.

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

    validate_rule_count(operation, rule_count)
    validate_noise_mode_config(
        noise_mode,
        label_noise,
        operation=operation,
        rule_count=rule_count,
        is_s5=False,
        label_mode=label_mode,
    )
    num_classes = category_label_num_classes(
        label_mode, label_mod, p=p, rule_count=rule_count,
    )

    all_pairs = domain_fn(p)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    if max_train_samples is not None and max_train_samples < len(train_pairs):
        train_pairs = train_pairs[:max_train_samples]

    x_train, y_train, ve = _encode_integer_pairs(
        train_pairs,
        op_fn,
        p,
        input_format,
        label_mode=label_mode,
        label_mod=label_mod,
        operation=operation,
        rule_count=rule_count,
    )
    x_val, y_val, _ = _encode_integer_pairs(
        val_pairs,
        op_fn,
        p,
        input_format,
        label_mode=label_mode,
        label_mod=label_mod,
        operation=operation,
        rule_count=rule_count,
    )
    _, _, base_vocab = _build_vocab_integer(p)
    vocab_size = base_vocab + ve

    apply_train_label_corruption(
        y_train,
        train_pairs,
        operation=operation,
        p=p,
        rule_count=rule_count,
        num_classes=num_classes,
        label_noise=label_noise,
        label_noise_sym=label_noise_sym,
        noise_mode=noise_mode,
        noise_fixed_target=noise_fixed_target,
        noise_fixed_backup=noise_fixed_backup,
        generator=_noise_generator(seed),
    )

    return (
        TensorDataset(x_train, y_train),
        TensorDataset(x_val, y_val),
        vocab_size,
        num_classes,
    )
