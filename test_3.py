import numpy as np


def failure_count_from_counts(counts, core_size, r):
    k = len(counts)
    m = int(np.sum(counts))

    C = np.concatenate(([0], np.cumsum(counts)))
    total_fail = 0

    for j in range(1, k + 1):
        left_rank = C[j - 1] + 1
        right_rank = C[j]

        if left_rank > right_rank:
            continue

        a = max(1, (j - 1) * core_size - r + 1)
        b = min(m, j * core_size + r)

        success_j = max(0, min(right_rank, b) - max(left_rank, a) + 1)
        total_fail += counts[j - 1] - success_j

    return int(total_fail)

def estimate_prob_by_sampling_values(n, k, core_size, r, q, num_trials=100_000, seed=None):
    rng = np.random.default_rng(seed)
    m = k * core_size
    success = 0

    edges = np.linspace(0.0, n, k + 1)

    for _ in range(num_trials):
        x = rng.uniform(0.0, n, size=m)
        counts, _ = np.histogram(x, bins=edges)
        f = failure_count_from_counts(counts, core_size, r)
        if f <= q:
            success += 1

    return success / num_trials


n = 1
k = 10
core_size = 100
r = 3
q = 0

n, k, core_size, r, q = int(n), int(k), int(core_size), int(r), int(q)

# for q in range(0, 10):
#     print(q, f"{estimate_prob_by_sampling_values(n, k, core_size, r, q, num_trials=1_000_000, seed=None):.2e}")


n = 1
k = 100_000
core_size = 1
r = 1_000
q = 0

n, k, core_size, r, q = int(n), int(k), int(core_size), int(r), int(q)


print(q, f"{estimate_prob_by_sampling_values(n, k, core_size, r, q, num_trials=100_000, seed=None):.2e}")