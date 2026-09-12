#!/usr/bin/env python3
"""Project what straggler handling would buy, on real per-chunk step counts.

Cost model (GB300, measured): a wave step costs step(B) = 0.764 + 0.0496*B ms
for B live items -- 76% of it fixed, which is why narrowing buys less than the
item-count reduction suggests. Re-forming a runtime costs ~12 ms (construction
plus the first step's graph capture).

Three policies, all on the same real step counts:
  current   groups of W, each running until its slowest member finishes
  narrow    one re-form per group, at the self-amortising point
  refill    keep W lanes busy by admitting the next pending chunk (continuous
            batching) -- an upper bound, since it ignores the ring redesign
"""
import sys

A, B_COEF, REFORM = 0.764, 0.0496, 12.0


def step_cost(live):
    return A + B_COEF * max(1, live) if live > 0 else 0.0


def groups(n, width):
    """Chunk 0 alone, then groups of `width`, matching the scheduler."""
    g = [[0]]
    at = 1
    while at < n:
        g.append(list(range(at, min(at + width, n))))
        at += width
    return g


def cost_current(steps, width):
    return sum(max(steps[i] for i in g) * step_cost(len(g)) for g in groups(len(steps), width))


def cost_perfect(steps, width):
    """Retire every item the moment it finishes. Unachievable bound."""
    total = 0.0
    for g in groups(len(steps), width):
        s = sorted(steps[i] for i in g)
        for t in range(max(s)):
            live = sum(1 for x in s if x > t)
            total += step_cost(live)
    return total


def cost_narrow(steps, width):
    """One re-form per group, at the first step where the waste already paid for it."""
    total = 0.0
    for g in groups(len(steps), width):
        s = sorted(steps[i] for i in g)
        W, T = len(g), max(s)
        best, fired = T * step_cost(W), False
        for t in range(1, T):
            live = sum(1 for x in s if x > t)
            if live == 0 or live == W:
                continue
            gain = step_cost(W) - step_cost(live)
            if gain <= 0 or t * gain < REFORM:   # self-amortising: waste must cover the re-form
                continue
            c = t * step_cost(W) + (T - t) * step_cost(live) + REFORM
            if c < best:
                best, fired = c, True
            break
        total += best
    return total


def cost_refill(steps, width):
    """Continuous batching: cost tracks total item-steps, not the slowest member."""
    total_item_steps = sum(steps)
    full, rem = divmod(total_item_steps, width)
    return full * step_cost(width) + (step_cost(rem) if rem else 0.0)


for path, width in [(p, 32) for p in sys.argv[1:]]:
    steps = [int(l.split()[1]) for l in open(path)]
    n = len(steps)
    cur = cost_current(steps, width)
    print(f"\n=== {path}  {n} chunks, width {width} ===")
    print(f"    steps: mean {sum(steps)/n:.0f}  max {max(steps)}  ratio {max(steps)/(sum(steps)/n):.2f}")
    for name, fn in [("current", cost_current), ("narrow (1 re-form)", cost_narrow),
                     ("perfect retire", cost_perfect), ("refill (bound)", cost_refill)]:
        c = fn(steps, width)
        print(f"    {name:20s} {c/1000:7.3f} s   {'--' if name=='current' else f'{(cur-c)/cur*100:+5.1f}%'}")
