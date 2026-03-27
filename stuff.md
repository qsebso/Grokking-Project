# Investigating Failure Modes in Multi-Rule Modular Arithmetic Learning

## Objective

The goal of this project was to understand why a transformer model can successfully learn individual modular arithmetic operations (e.g., addition or multiplication mod *p*), but fails when tasked with learning a conditional composition of two such rules (e.g., `add_or_mul`).

---

## Experimental Setup

### Base Task

The model is trained on inputs of the form:

(a, op, b, =)

with targets defined as:

c = op(a, b) mod p

where `p = 97`.

---

## Phase 1: Single-Rule Learning

### Experiments

- `add`
- `mul`

### Result

- Both tasks achieved near-perfect training and validation accuracy.
- Grokking behavior was observed.

### Interpretation

The model is capable of learning modular arithmetic functions in isolation.

---

## Phase 2: Multi-Rule Task

### Experiment

- `add_or_mul`:
  - If `b` is odd → addition
  - If `b` is even → multiplication

### Result

- Training and validation accuracy plateaued significantly below 100%.
- No grokking observed.

### Initial Hypothesis

The failure might be due to:

- increased task complexity
- interference between rules
- insufficient capacity or optimization issues

---

## Phase 3: Disjoint Label Encoding

### Modification

Targets were redefined to remove overlap between rules:

y = rule_id * p + c_local

- `rule_id ∈ {0,1}`
- `c_local ∈ {0,...,p-1}`
- Total classes: `2p = 194`

### Result

- Performance remained capped (~0.87 train, ~0.63 validation).
- No meaningful improvement.

### Interpretation

Label overlap was not the root cause.

---

## Phase 4: Factorized Output Model

### Modification

Instead of predicting a single class, the model was modified to predict:

- `rule_id` (2 classes)
- `c_local` (97 classes)

Three modes were tested:

- `rule_only`
- `c_only`
- `joint`

### Result

- `rule_only` plateaued at ~0.63 validation accuracy
- `c_only` plateaued at ~0.26 validation accuracy
- `joint` reflected the combined limitations

### Interpretation

The issue is not caused by flattening the output space. The model struggles with both components independently.

---

## Phase 5: Isolating the Routing Problem

### Key Experiment

--factor_mode rule_only

### Observation

Despite being a simple binary classification task (based on parity of `b`), performance remained capped.

### Controls Tested

- Model size
- Learning rate
- Weight decay
- Input format

### Result

- All configurations converged to the same suboptimal solution.

### Interpretation

The failure is not due to optimization or capacity. The model consistently learns an approximate but incorrect partition of the input space.

---

## Phase 6: Injecting Parity Explicitly

### Modification

Two new input formats were introduced:

1. Full input + parity:

[a, op, b, b mod 2, =]

1. Parity only:

[a, op, b mod 2, =]

### Results

#### Parity Only

- `rule_only` → ~100% accuracy (train and validation)

#### Full Input + Parity

- `rule` head → ~100% accuracy
- `c_local`:
  - Train → ~100%
  - Validation → ~33% (improving slowly)

### Interpretation

Providing parity directly resolves the routing problem entirely. This confirms that the model was unable to infer parity from token embeddings.

---

## Key Findings

### 1. Failure to Learn Parity

The model fails to infer:

rule_id = 1 - (b mod 2)

from raw token representations of `b`.

This is the primary bottleneck in the multi-rule task.

---

### 2. Routing as the Primary Failure Point

Without parity:

- Rule selection is unreliable
- Downstream computation fails

With parity:

- Rule selection becomes perfect
- Downstream computation improves significantly

---

### 3. Conditional Generalization Is Delayed, Not Absent

Even with perfect routing:

- The model initially struggles to generalize the conditional arithmetic function
- Training accuracy reaches ~100% early
- Validation accuracy remains low for several thousand epochs

> Validation → initially low (~0.3), but increases sharply later with extended training

However, with longer training, validation accuracy rises sharply and eventually approaches ~100%. This demonstrates that the model is capable of learning and generalizing the full conditional arithmetic function, but only after a period of delayed generalization (grokking).

This indicates that the issue is not an inherent inability to generalize compositional functions under conditional structure, but rather that such generalization occurs later in training once sufficient internal structure has been learned.

---

## Final Conclusion

The failure of the original `add_or_mul` task is not due to an inability to learn multiple arithmetic rules or to generalize a conditional arithmetic function. Instead, the primary bottleneck is the model’s inability to infer the rule-selection condition—namely, the parity of `b`—from raw token embeddings.

Once parity is explicitly provided, the model learns the routing function perfectly and is able to eventually grok the full two-rule conditional task, reaching near-perfect validation accuracy after sufficient training.

This shows that the core limitation lies in discovering discrete structure from the input representation, rather than in conditional computation itself.

---

## Implications

This work highlights two important limitations of transformer-based models:

1. **Lack of inductive bias for discrete structure**
  Models do not naturally discover properties such as parity from arbitrary token embeddings.
2. **Difficulty with compositional generalization**
  Even when components are individually learnable, combining them under conditional logic introduces additional challenges.

---

## Future Work

Potential next steps include:

- Testing alternative representations of inputs (e.g., binary encoding of numbers)
- Exploring architectures with stronger inductive bias for modular structure
- Evaluating simpler conditional tasks (e.g., `add_or_sub`)
- Increasing training data coverage (`train_frac`)
- Investigating mechanisms for explicit feature disentanglement

