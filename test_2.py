import math
from functools import lru_cache

def binom_pmf(x, n, p):
    return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

def interval_failures(j, s_prev, x, core_size, r, m):
    """
    j is 1-indexed bin number
    s_prev = cumulative count before this bin
    x = count in this bin
    """
    left_rank = s_prev + 1
    right_rank = s_prev + x

    if x == 0:
        return 0

    a = max(1, (j - 1) * core_size - r + 1)
    b = min(m, j * core_size + r)

    success_j = max(0, min(right_rank, b) - max(left_rank, a) + 1)
    return x - success_j

def exact_prob_at_most_q_fail_dp(k, core_size, r, q):
    m = k * core_size

    @lru_cache(None)
    def dp(j, s, f):
        """
        processed first j-1 bins
        s = cumulative count used so far
        f = failures so far
        returns probability
        """
        if f > q:
            return 0.0
        if j == k + 1:
            return 1.0 if s == m else 0.0

        remaining = m - s
        bins_left = k - j + 1
        p = 1.0 / bins_left

        total = 0.0
        for x in range(remaining + 1):
            # conditional distribution of count in current bin
            prob_x = binom_pmf(x, remaining, p)
            add_f = interval_failures(j, s, x, core_size, r, m)
            total += prob_x * dp(j + 1, s + x, f + add_f)

        return total

    return dp(1, 0, 0)



m = 1000
k = 10
max_r = 3
q = 0

core_size = m//k

for r in range(0, max_r):
    print((m, core_size, r, q), f"{exact_prob_at_most_q_fail_dp(k, core_size, r, q):.2e}")

r = 3
max_tolerated_failure = 10
for q in range(0, max_tolerated_failure):
    print((m, core_size, r, q), f"{exact_prob_at_most_q_fail_dp(k, core_size, r, q):.2e}")