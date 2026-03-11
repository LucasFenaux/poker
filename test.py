import math


def f(n, k, m):
    numerator = math.pow(math.comb(math.floor(n/k), math.floor(m/k)), k)
    # print(numerator)
    denominator = math.comb(n, m)
    # print(denominator)
    # print(numerator/denominator)
    return numerator/denominator

def g(n, k, m):
    assert n % k == 0
    assert m % k == 0
    numerator = math.pow(math.comb(int(n/k), int(m/k)), k)
    # print(numerator)
    denominator = math.comb(n, m)
    # print(denominator)
    # print(numerator/denominator)
    return numerator/denominator



# print(f"{f(20, 2, 10)}")
# for n in [10, 50, 100]:
#     for k in [2, 5, 10]:
#         for m in [5, 10, 20]:
#             if n > k and n > m and m > k:
#                 n = (n // k) * k
#                 m = (m // k) * k
#                 print(f"{(n,k,m)} {g(n, k, m)}")



n = 1000
m = 100
k = 10
print(g(n, k, m))
# print(f"{f(20, 5, 10)}")
# print(f"{g(20, 5, 10)}")


import math
from functools import lru_cache

def failure_count(counts, core_size, r):
    """
    counts[j] = N_{j+1}
    core_size = c
    server stored bin size = c + 2r for interior bins
    """
    k = len(counts)
    m = sum(counts)

    C = [0]
    for x in counts:
        C.append(C[-1] + x)

    total_fail = 0

    for j in range(1, k + 1):
        left_rank = C[j - 1] + 1
        right_rank = C[j]

        # empty interval
        if left_rank > right_rank:
            continue

        a = max(1, (j - 1) * core_size - r + 1)
        b = min(m, j * core_size + r)

        overlap_len = max(0, min(right_rank, b) - max(left_rank, a) + 1)
        success_j = overlap_len
        fail_j = counts[j - 1] - success_j
        total_fail += fail_j

    return total_fail


def multinomial_prob(counts):
    m = sum(counts)
    k = len(counts)
    out = math.factorial(m) / (k ** m)
    for x in counts:
        out /= math.factorial(x)
    return out


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for x in range(total + 1):
        for rest in compositions(total - x, parts - 1):
            yield (x,) + rest


def prob_at_most_q_fail(k, core_size, r, q):
    m = k * core_size
    prob = 0.0
    for counts in compositions(m, k):
        f = failure_count(counts, core_size, r)
        if f <= q:
            prob += multinomial_prob(counts)
    return prob

m = 50
k = 5
max_r = 3
q = 0

core_size = m//k

for r in range(0, max_r):
    print((m, core_size, r, q), f"{prob_at_most_q_fail(k, core_size, r, q):.2f}")

r = 3
max_tolerated_failure = 10
for q in range(0, max_tolerated_failure):
    print((m, core_size, r, q), f"{prob_at_most_q_fail(k, core_size, r, q):.2f}")

