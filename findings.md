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