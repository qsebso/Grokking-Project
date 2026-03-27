Clean ranking of what you tested
Strongest
(a+b)%3 with mixed structured branches → 0.915
Middle
(a-b)%3 with a+b, a+2b, a*b → 0.831
(a-b)%3 with a+b, a-b, a*b → 0.830
Weaker
a%3 with a+b, a+2b, a*b → 0.807
b%3 with a+b, a+2b, a*b → 0.805
Weakest
(a-b)%3 with a-b, a-2b, a*b → 0.756

In 3-way piecewise modular tasks, changing the branch operations had limited impact when the routing function was fixed. In contrast, changing the routing function substantially changed performance. The best results were consistently obtained with routing based on (a+b) mod 3, while one-coordinate routing (a mod 3, b mod 3) and difference-based routing ((a-b) mod 3) performed worse. This suggests that generalization is driven more by the geometry and algebraic coherence of the partition than by the exact branch rules themselves.



Routing by (a+b) mod k is consistently best.
Single-variable routing is worse.
Difference-based routing is worse.
4-way can outperform 3-way if the branch family is coherent.
The best branch families seem to mix variation with structure, not just random rule soup.





categorical 
PS C:\Users\Quinn\Desktop\ML Project\Grokking-Project> python -m experiments.train_category --operation add_or_mul --label_mode c_mod  --label_mod 2 --log_every 1
train=4,704  val=4,705  vocab=99  classes=2  (label_mod=2)
Device: cuda
Categorical task: label_mode=c_mod  label_mod=2  num_classes=2
Parameters: 409,474
  Log format: cls% uses class ids 0..K-1 left-to-right (whole %); after '|' is the branch_metric split on inputs (also %).
C:\Users\Quinn\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\functional.py:5504: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)   
  attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
  Epoch    10 | Train 0.489 | Val 0.491 | Loss 0.7465 / 0.7493
      cls 0..1 (%)  tr 0/100  |  val 0/100  |  b_parity  tr odd/even 50%/48%  val 49%/49%
  Epoch    20 | Train 0.513 | Val 0.511 | Loss 0.6925 / 0.6998
      cls 0..1 (%)  tr 99/1  |  val 99/1  |  b_parity  tr odd/even 51%/52%  val 51%/51%
  Epoch    30 | Train 0.591 | Val 0.557 | Loss 0.6710 / 0.6832
      cls 0..1 (%)  tr 52/66  |  val 48/64  |  b_parity  tr odd/even 64%/54%  val 62%/50%
  Epoch    40 | Train 0.618 | Val 0.583 | Loss 0.6572 / 0.6783
      cls 0..1 (%)  tr 55/69  |  val 51/66  |  b_parity  tr odd/even 69%/54%  val 67%/50%
  Epoch    50 | Train 0.647 | Val 0.603 | Loss 0.6346 / 0.6661
      cls 0..1 (%)  tr 63/66  |  val 58/63  |  b_parity  tr odd/even 74%/56%  val 70%/51%
  Epoch    60 | Train 0.679 | Val 0.629 | Loss 0.5909 / 0.6385
      cls 0..1 (%)  tr 66/70  |  val 60/66  |  b_parity  tr odd/even 79%/57%  val 76%/51%
  Epoch    70 | Train 0.754 | Val 0.674 | Loss 0.4848 / 0.5686
      cls 0..1 (%)  tr 76/75  |  val 66/68  |  b_parity  tr odd/even 87%/64%  val 84%/51%
  Epoch    80 | Train 0.800 | Val 0.695 | Loss 0.4019 / 0.5627
      cls 0..1 (%)  tr 74/86  |  val 62/77  |  b_parity  tr odd/even 92%/68%  val 87%/53%
  Epoch    90 | Train 0.849 | Val 0.697 | Loss 0.3251 / 0.6241
      cls 0..1 (%)  tr 80/90  |  val 63/77  |  b_parity  tr odd/even 93%/76%  val 85%/56%
  Epoch   100 | Train 0.888 | Val 0.698 | Loss 0.2485 / 0.7089
      cls 0..1 (%)  tr 85/93  |  val 65/75  |  b_parity  tr odd/even 94%/83%  val 82%/58%
  Epoch   110 | Train 0.913 | Val 0.696 | Loss 0.1817 / 0.8471
      cls 0..1 (%)  tr 95/87  |  val 72/67  |  b_parity  tr odd/even 94%/89%  val 80%/60%
  Epoch   120 | Train 0.923 | Val 0.697 | Loss 0.1628 / 0.9836
      cls 0..1 (%)  tr 99/86  |  val 74/65  |  b_parity  tr odd/even 93%/91%  val 79%/61%
  Epoch   130 | Train 0.932 | Val 0.696 | Loss 0.1176 / 1.1295
      cls 0..1 (%)  tr 93/93  |  val 70/70  |  b_parity  tr odd/even 93%/93%  val 78%/62%
  Epoch   140 | Train 0.934 | Val 0.696 | Loss 0.1108 / 1.2320
      cls 0..1 (%)  tr 98/88  |  val 70/69  |  b_parity  tr odd/even 94%/93%  val 78%/62%
  Epoch   150 | Train 0.934 | Val 0.699 | Loss 0.1172 / 1.3046
      cls 0..1 (%)  tr 100/87  |  val 72/68  |  b_parity  tr odd/even 93%/94%  val 78%/62%
  Epoch   160 | Train 0.935 | Val 0.699 | Loss 0.1017 / 1.3454
      cls 0..1 (%)  tr 88/100  |  val 68/72  |  b_parity  tr odd/even 94%/93%  val 78%/62%
  Epoch   170 | Train 0.935 | Val 0.699 | Loss 0.0983 / 1.3681
      cls 0..1 (%)  tr 88/100  |  val 69/71  |  b_parity  tr odd/even 94%/93%  val 78%/62%
  Epoch   180 | Train 0.936 | Val 0.697 | Loss 0.1064 / 1.3792
      cls 0..1 (%)  tr 100/87  |  val 71/68  |  b_parity  tr odd/even 93%/94%  val 78%/62%
  Epoch   190 | Train 0.936 | Val 0.697 | Loss 0.1140 / 1.4117
      cls 0..1 (%)  tr 87/100  |  val 67/73  |  b_parity  tr odd/even 94%/93%  val 79%/61%
  Epoch   200 | Train 0.936 | Val 0.696 | Loss 0.1019 / 1.3934
      cls 0..1 (%)  tr 87/100  |  val 68/72  |  b_parity  tr odd/even 94%/93%  val 78%/61%
  Epoch   210 | Train 0.936 | Val 0.696 | Loss 0.1040 / 1.4069
      cls 0..1 (%)  tr 100/87  |  val 71/68  |  b_parity  tr odd/even 93%/94%  val 78%/61%
  Epoch   220 | Train 0.936 | Val 0.694 | Loss 0.0958 / 1.3851
      cls 0..1 (%)  tr 88/100  |  val 68/71  |  b_parity  tr odd/even 94%/93%  val 78%/61%
  Epoch   230 | Train 0.936 | Val 0.696 | Loss 0.1026 / 1.4300
      cls 0..1 (%)  tr 100/87  |  val 71/68  |  b_parity  tr odd/even 93%/94%  val 78%/61%

waht does this tell you

This is actually a really clean result. No guessing needed — here’s exactly what’s happening.
 Big picture

You ran:

add_or_mul + c_mod 2

So:

label = (c % 2)

Which means:

ADD → parity of (a + b)
MUL → parity of (a * b)
What the model figured out

Look at this:

Train → 0.936
Val   → ~0.696

And more importantly:

b_parity:
train odd/even → 94% / 93%
val   odd/even → 78% / 61%
This is the key insight
The model learned ONE branch well
Odd branch → ~78% val
Even branch → ~61% val

That’s a big gap

Why?

Because parity behaves VERY differently for add vs mul.

ADD parity:
(a + b) % 2 = a%2 XOR b%2

→ simple linear-ish structure
→ easy to learn

MUL parity:
(a * b) % 2 = 1 only if both odd

→ AND relationship
→ harder pattern

What your model did

It learned:

“I can partially solve parity without fully knowing which rule is used.”

It is NOT doing:

detect rule → compute → mod

It is doing:

learn shortcuts directly on (a, b)
Evidence in your logs
Early epochs:
Epoch 20 → cls 99/1

→ collapsed to predicting ONE class
→ classic lazy solution

Later:
Epoch 70+ → balanced predictions

→ model starts picking up structure

Final:
Train → ~94%
Val   → stuck ~70%
Loss increasing on val

That means:

memorization happening
generalization plateau
no grokking
The most important signal

This line:

val odd/even → 78% / 61%

That tells you:

The model is NOT learning the unified function
It’s learning branch-dependent heuristics

Translate this into plain English

Your dataset says:

sometimes use add
sometimes use mul
then take parity

But your model says:

I don’t fully know the rule,
but I can guess parity pretty well from patterns in (a,b)



It’s not useless. It’s actually one of the more useful results you’ve gotten, because it separates “the model can learn exact modular outputs categorically” from “the model can handle a mixed rule.”

What you’ve shown so far is:

add groks in c mode
so categorical exact-output prediction is a legit setup, not a broken one
add_or_mul does not behave the same way in c mode
so the problem is the two-rule structure, not just output format

That is a real finding.


What your experiment now says

You’ve now shown three important things:

1. add with exact categorical output can memorize and grok

So categorical exact-output learning works.

2. add_or_mul with exact categorical output does not memorize

So the failure is not caused by compression like mod 2.

3. add_or_mul still does not memorize even after removing overlap

So the issue is not label collision. It is the mixed-rule structure itself.

That’s actually a clean finding.

Blunt interpretation

Your current two-rule problem is probably just too hard for this model/setup in the exact-output regime.

Not impossible in principle. Just too hard under:

this capacity
this optimization
this weight decay
this representation