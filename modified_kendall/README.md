# Modified Kendall rank-order association test — Python implementation

This is a tested Python translation of the supplied `functions.R` and
`seqn_agg_func.R`, plus a user-facing implementation of the test in Zheng and
Lo (2006).

## What the method does

For two 1-based rankings of the same `n` objects, smaller ranks are better. At
truncation value `k`, the paper replaces every rank `r` with `min(r, k)` and
counts concordant (“agreement”) pairs. The count is standardized using the
closed-form null mean and variance from the paper. This emphasizes repeatable
ordering near the top while tying the uninformative lower ranks.

The paper/R convention has an important detail: `min(r, k)` leaves ranks
`1, ..., k-1` distinct and ties ranks `k, ..., n`.

## Installation

Only NumPy is required for the core module. Matplotlib is required for the
simulation example.

```bash
pip install numpy matplotlib
```

## Basic use with existing ranks

```python
import numpy as np
from modified_kendall import modified_kendall_test

rank_x = np.array([1, 4, 2, 5, 3], dtype=float)
rank_y = np.array([1, 3, 2, 5, 4], dtype=float)

result = modified_kendall_test(rank_x, rank_y, k=4)
print(result.z_statistic, result.pvalue)
```

The default p-value is the one-sided upper-tail normal approximation used in
the paper. For small `n`/`k`, or for original rankings containing ties, use a
permutation p-value:

```python
result = modified_kendall_test(
    rank_x,
    rank_y,
    k=4,
    permutations=10_000,
    random_state=123,
)
```

## Use with raw scores

```python
from modified_kendall import modified_kendall_from_scores

result = modified_kendall_from_scores(
    scores_x,
    scores_y,
    k=50,
    descending=True,   # larger score = better object
)
```

## Statistic over every truncation value

```python
from modified_kendall import sequential_agreement

seq = sequential_agreement(rank_x, rank_y)
# seq.k, seq.agreement, seq.null_mean, seq.null_variance, seq.z_statistic
```

Unlike the original O(n²) incremental R code, the sequence implementation uses
an O(n log n) dominance-count identity.

## R-compatible names

The module also exports:

- `f_agg`
- `f_choose`
- `f_theory`
- `f_disagg`
- `f_tie`
- `f_seqn_agg`

## Run the supplied-style simulation

```bash
python example_modified_kendall.py
```

## Run tests

```bash
python test_modified_kendall.py
```

The tests compare optimized counts with direct O(n²) counts and verify the
paper’s null mean and variance by exact permutation enumeration for small `n`.
