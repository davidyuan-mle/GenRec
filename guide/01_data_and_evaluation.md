# Part 1: Data and Evaluation

> **After this part you should be able to explain:** where our user sequences come from, why we filter the data, how one user's history is split into training / validation / test, and exactly how Recall@K, NDCG@K and MRR turn a top-K list into a score.

Every later part builds on this one: a model is only as good as the way we measure it.

---

## 1.1 The raw data: Amazon reviews as a stand-in for purchases

We use the **Amazon Beauty** dataset (2014 release, from Stanford SNAP). It contains product reviews. Each review gives us three things:

| Field | Example | Used as |
|---|---|---|
| `reviewerID` | `A1YJEY40YUW4SE` | the **user** |
| `asin` | `B000W30RP8` | the **item** (Amazon's product ID) |
| `unixReviewTime` | `1369008000` (2013-05-20) | the **time**, used to put items in order |

A review is not a purchase, but people mostly review things they bought, so "user reviewed item at time $t$" is used as "user interacted with item at time $t$". This is the standard benchmark for Semantic-ID papers (TIGER evaluates on it), which is why we use it.

There's also a **metadata** file with each product's title, brand, categories, price and description. Part 3 turns that text into vectors.

---

## 1.2 Cleaning: the 5-core filter

### Intuition

Most users write one review and disappear. Most products get one or two reviews. A sequence of length 1 has no "next item" to predict, and an item seen once can't be learned. So we keep only users and items with **at least 5 interactions**. This is called the **5-core**.

The catch: removing users can push items below 5 interactions, and removing those items can push users below 5. So we repeat until nothing changes.

### The math

Let $D$ be the set of (user, item) interactions. Define the counts $n_u(D)$ = interactions of user $u$ and $n_i(D)$ = interactions of item $i$. The $k$-core is the largest subset $D_k \subseteq D$ such that

$$
n_u(D_k) \ge k \quad  \text{for every user in } D_k, \qquad n_i(D_k) \ge k \quad  \text{for every item in } D_k
$$

The loop "drop everything below $k$, recount, repeat" finds it.

### Worked example ($k = 2$)

| Round | Interactions | Problem | Action |
|---|---|---|---|
| start | u1–a, u1–b, u2–a, u2–b, u3–c | u3 has 1 interaction | drop u3 |
| 1 | u1–a, u1–b, u2–a, u2–b | item c now has 0, but it's already gone with u3 | nothing left to drop |
| done | 4 interactions, users {u1, u2}, items {a, b} | | |

In this small case one round was enough. A trickier case: if u3 had also reviewed item `a`, dropping u3 would lower `a`'s count, which could push `a` below $k$, which could push u1 below $k$, and so on. That's why the code loops.

### In our code and data

- [amazon.py](../src/genrec/data/amazon.py): `kcore_filter` runs the loop. `build_dataset` then sorts each user's items by time and renumbers items as $1 \dots N$, keeping $0$ for padding (Part 2).
- [test_data.py](../tests/test_data.py): `test_kcore_filter_is_iterative` checks the "removal causes more removal" case.

After filtering, our dataset is:

| | Value |
|---|---|
| Users | 22,363 |
| Items | 12,101 |
| Interactions | 198,502 |
| Average / median history length | 8.88 / 6 items |
| Users with 7 or fewer items | 62.9% |
| Density (share of all user–item pairs that happened) | 0.073% |
| Top 1% / top 10% of items cover | 10.5% / 42.8% of interactions |

Two facts to remember: **histories are short** (most users have 5–7 items), and **popularity is skewed** (a few items get a big share of the interactions). Both shape everything later.

---

## 1.3 Splitting: leave-one-out

### Intuition

To test a recommender honestly, we must hide some answers from it during training. For sequences, the standard approach is **leave-one-out**: for each user, hide their **last** item for testing and their **second-to-last** for validation (tuning). Train on everything before that.

### The math

For a user with history $(i_1, i_2, \dots, i_n)$:

| Split | Model sees (input) | Must predict (target) |
|---|---|---|
| train | $(i_1, \dots, i_{n-2})$ | each next item inside this prefix |
| validation | $(i_1, \dots, i_{n-2})$ | $i_{n-1}$ |
| test | $(i_1, \dots, i_{n-1})$ | $i_n$ |

The 5-core guarantees $n \ge 5$, so every user has at least 3 training items.

### Worked example: user #17 from Part 0

| Position | Item |
|---|---|
| 1 | Leave-In Conditioner |
| 2 | Miracle Oil |
| 3 | Miracle Shampoo |
| 4 | Deep Conditioner |
| 5 | Leave-In Potion |

- **Train** on (Leave-In Conditioner, Oil, Shampoo): the model may learn "after Leave-In Conditioner comes Oil" and "after Oil comes Shampoo".
- **Validation:** given (Leave-In Conditioner, Oil, Shampoo), predict **Deep Conditioner**.
- **Test:** given all of the first four, predict **Leave-In Potion**.

Notice that at test time the model is allowed to see the validation item (Deep Conditioner) in the history. It just never *trained* on predicting it.

### Two subtleties worth knowing

1. **Validation is for decisions, test is for reporting.** We pick settings (when to stop training, kNN's decay, …) using validation scores only. The test split is touched once, at the end. Otherwise we would be tuning to the test set and our reported numbers would be too optimistic.
2. **Leave-one-out mixes time periods.** User A's test item might be from 2010 and user B's from 2014. A real deployment can only train on the past and predict the future for everyone at once. This is a known weakness of the benchmark, accepted because everyone uses it, which makes results comparable. It may be why our popularity and SASRec scores drop from validation to test (seen in Parts 2–3).

**In our code:** [sequences.py](../src/genrec/data/sequences.py), `leave_one_out`. Notebook 01 also found that **33 test targets never appear in any training sequence**: items so rare that their only appearances are as someone's final item. An ID-based model like SASRec can never rank these well, whereas a text-based model at least has their description.

---

## 1.4 Evaluation: scoring a top-K list

### Intuition

For each test user, the model returns its top-K items, best first. There's exactly **one** correct answer: the item they really chose next. A good metric should reward:

- the correct item **being in the list** at all, and
- the correct item being **near the top** of the list.

We use three metrics. Each is computed per user and then averaged over all users.

### The math

Let $\text{rank}$ be the position of the true item in the model's list (1 = top), or "not found".

**Recall@K** (also called Hit Rate@K): was it in the top K?

$$
\text{Recall@}K = \begin{cases} 1 & \text{if rank} \le K \cr  0 & \text{otherwise} \end{cases}
$$

**NDCG@K** (Normalized Discounted Cumulative Gain): like Recall, but a hit lower in the list counts for less. With only one correct item, the formula simplifies to

$$
\text{NDCG@}K = \begin{cases} \dfrac{1}{\log_2(\text{rank} + 1)} & \text{if rank} \le K \cr  0 & \text{otherwise} \end{cases}
$$

The "discount" $\frac{1}{\log_2(\text{rank}+1)}$ is 1 at rank 1, 0.63 at rank 2, 0.5 at rank 3, 0.39 at rank 5, and 0.29 at rank 10. It falls quickly at first and then slowly.

> **Why does it simplify?** The general NDCG divides the "gain" of your list by the gain of a perfect list (called IDCG). With one relevant item, a perfect list puts it at rank 1, so IDCG $= 1/\log_2 2 = 1$, and dividing by 1 changes nothing.

**MRR** (Mean Reciprocal Rank): one over the rank.

$$
\text{MRR} = \begin{cases} \dfrac{1}{\text{rank}} & \text{if found} \cr  0 & \text{otherwise} \end{cases}
$$

It punishes low ranks harder than NDCG: rank 2 is worth 0.5, and rank 10 is worth 0.1.

### Worked example

Three users, each with a top-5 list:

| User | True next item | Model's top 5 | Rank |
|---|---|---|---|
| 1 | conditioner | shampoo, **conditioner**, hair oil, comb, mask | 2 |
| 2 | top coat | polish A, polish B, file, remover, **top coat** | 5 |
| 3 | perfume | lipstick, mascara, liner, blush, gloss | not found |

| User | Recall@1 | Recall@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| 1 | 0 | 1 | $1/\log_2 3 = 0.631$ | $1/2 = 0.500$ |
| 2 | 0 | 1 | $1/\log_2 6 = 0.387$ | $1/5 = 0.200$ |
| 3 | 0 | 0 | 0 | 0 |
| **Average** | **0.000** | **0.667** | **0.339** | **0.233** |

Reading the averages: two thirds of users got their item in the top 5 (Recall@5 = 0.667). But the hits weren't at the very top, so NDCG@5 is much lower (0.339).

### How to read our numbers

Our catalog has 12,101 items. A model that guesses randomly would get Recall@10 $\approx 10 / 12{,}101 \approx 0.0008$. So when SASRec gets **Recall@10 = 0.0753**, it puts the true next item in its top 10 for about 1 user in 13. That is roughly **90× better than random**, even though the number looks small.

Values around 0.03–0.08 for Recall@10 are normal on this benchmark. Compare models with each other, not with 1.0.

### In our code

- [metrics.py](../src/genrec/metrics.py): `recall_at_k`, `ndcg_at_k`, `mrr`, and `evaluate`, which computes all of them for $K \in \lbrace 5, 10, 20\rbrace $.
- Every model has the same method, `recommend(histories, k)`, which returns top-K lists. That's what makes all models comparable with the same `evaluate` call.
- One detail: models return only their top 20 items, so our MRR is really "MRR@20": a true item at rank 21 or lower counts as 0. That's standard practice, and the difference is tiny because $1/21$ is already small.
- [scripts/train.py](../scripts/train.py) writes `metrics.json` for both validation and test into `experiments/<run>/`, and [RESULTS.md](../experiments/RESULTS.md) collects the test numbers.

---

## 1.5 Check yourself

<details>
<summary><b>Q1.</b> A user has history (a, b, c, d, e, f). What does the model see and predict for validation and for test?</summary>

Validation: sees (a, b, c, d), predicts e. Test: sees (a, b, c, d, e), predicts f. Training uses only (a, b, c, d).
</details>

<details>
<summary><b>Q2.</b> Model X puts the true item at rank 1 for half the users and misses the rest. Model Y puts it at rank 3 for every user. Compare Recall@5, NDCG@5 and MRR.</summary>

X: Recall@5 = 0.5, NDCG@5 = 0.5 × 1 = 0.5, MRR = 0.5 × 1 = 0.5.
Y: Recall@5 = 1.0, NDCG@5 = $1/\log_2 4$ = 0.5, MRR = 1/3 ≈ 0.333.
Same NDCG, but Recall prefers Y and MRR prefers X. Different metrics reward different behaviour, which is why we report several.
</details>

<details>
<summary><b>Q3.</b> Why can't we choose the number of training epochs by looking at test scores?</summary>

Then the test set would be used to make a decision, and the reported test score would be biased upward: we would have picked whichever setting happened to fit the test users best. Decisions use validation; test is reported once.
</details>

<details>
<summary><b>Q4.</b> Why does the 5-core filter need a loop instead of a single pass?</summary>

Removing a user lowers the counts of the items they interacted with, which can push those items below the threshold. Removing those items lowers other users' counts, and so on. You repeat until no count changes.
</details>

---

**Next:** [Part 2, Baselines](02_baselines.md): the simplest possible recommender, and SASRec, the "score every item" model we're trying to beat.
