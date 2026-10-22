# Part 0: The Big Picture

> **After this part you should be able to explain, in two minutes:** what problem a recommender solves, the two very different ways to solve it ("score every item" vs. "generate the item's name"), what a Semantic ID is, and how the pieces of this repo fit together.

---

## 0.1 The problem: guess what comes next

Here is a real user from our Amazon Beauty data (user #17). These are their purchases, oldest first:

1. It's a 10 Keratin **Leave-In Conditioner**
2. It's a 10 Miracle **Oil** Plus Keratin
3. It's a 10 Miracle **Shampoo** Plus Keratin
4. It's a 10 Miracle **Deep Condit*ioner** Plus Kerat*in
5. It's a 10 Miracle **Leave-In Potion** Plus Keratin

If you only saw items 1–4, what would you guess for item 5? Probably "something else from the It's a 10 line". That guess is exactly what a **sequential recommender** tries to make.

### The math

We write a user's history as a list of item IDs:

$$
h = (i_1, i_2, \dots, i_n)
$$

and the catalog of all items as $\mathcal{I}$, with $N = |\mathcal{I}|$ items in total. The task is to estimate, for every item $i$ in the catalog,

$$
P(i_{n+1} = i \mid h)
$$

which reads: "the probability that the user's **next** item is $i$, given their history $h$". We then show the user the $K$ items with the highest probability: a **top-K list**.

We judge a recommender by how often the item the user *actually* chose next appears near the top of that list. Part 1 explains the exact scores (Recall@K, NDCG@K, MRR).

> **Key point.** Every model in this repo, from the dumbest to the fanciest, answers the same question: *given $h$, rank the catalog.* The models differ only in **how** they produce that ranking.

---

## 0.2 Approach 1: give every item a score (the classic way)

The traditional approach: learn one vector per item and one vector per user history, and score each item by how well the two vectors line up.

### The math

- Each item $i$ gets a learned vector $\mathbf{v}_i \in \mathbb{R}^d$ (a list of $d$ numbers, e.g. $d = 64$).
- A model reads the history $h$ and produces a vector $\mathbf{u}_h \in \mathbb{R}^d$ that summarizes "what this user wants next".
- The score of item $i$ is the dot product:

$$
s(h, i) = \mathbf{u}_h \cdot \mathbf{v}_i = \sum_{j=1}^{d} u_{h,j} \, v_{i,j}
$$

- To turn scores into probabilities, apply the **softmax**: exponentiate, then divide by the total, so the results are positive and add up to 1:

$$
P(i \mid h) = \frac{e^{s(h,i)}}{\sum_{i' \in \mathcal{I}} e^{s(h,i')}}
$$

Note the sum in the denominator: **it runs over every item in the catalog.**

### Worked example

A toy catalog of 3 items, with $d = 2$ so we can compute by hand:

| Item | $\mathbf{v}_i$ |
|---|---|
| shampoo | $(1, 0)$ |
| conditioner | $(0.5, 1)$ |
| nail polish | $(-1, 0.5)$ |

Suppose the model reads the history and outputs $\mathbf{u}_h = (1, 2)$. Each score is the dot product of $\mathbf{u}_h$ with that item's vector:

| Item | Score $s(h,i)$ | $e^{s}$ | $P(i \mid h)$ |
|---|---|---|---|
| shampoo | $1 \cdot 1 + 2 \cdot 0 = 1$ | $2.72$ | $2.72 / 15.90 = 0.17$ |
| conditioner | $1 \cdot 0.5 + 2 \cdot 1 = 2.5$ | $12.18$ | $12.18 / 15.90 = \mathbf{0.77}$ |
| nail polish | $1 \cdot (-1) + 2 \cdot 0.5 = 0$ | $1.00$ | $1.00 / 15.90 = 0.06$ |
| | | total $= 15.90$ | total $= 1.00$ |

Top-1 recommendation: **conditioner**. The geometry is the whole story: the history vector points in roughly the same direction as the conditioner vector.

### Where it struggles

This works well (it's what SASRec does in Part 2), but it has three weaknesses that matter at industrial scale:

1. **One vector per item means a giant table.** The table has $N \times d$ numbers. At $N = 12{,}101$ and $d = 64$ that's 0.8 million numbers, which is fine. At YouTube scale ($N$ in the billions) the table dominates the whole model.
2. **New items start knowing nothing.** A product added yesterday has a $\mathbf{v}_i$ that is still basically random, because nobody has bought it yet. This is the **cold-start** problem.
3. **Every item learns alone.** Our data has dozens of China Glaze nail-polish shades. Each one gets its own vector, learned only from its own handful of purchases, even though they are almost the same product.

---

## 0.3 Approach 2: generate the item's "name" (the generative way)

Here is the key idea of this whole repo.

> Instead of scoring all $N$ items, give every item a **short code made of a few small numbers**, chosen so that **similar items get similar codes**. Then train a model to **write out the code of the next item, one number at a time**, the same way a language model writes a sentence one word at a time.

That short code is called a **Semantic ID**. Think of a postal code: `94110` narrows from region (`9`) to area (`94`) to neighbourhood (`94110`). Two houses on the same street share most of the code. Semantic IDs do the same for products: the first number says roughly what kind of product it is, and later numbers narrow it down.

### A real example from our data

These are actual RQ-VAE Semantic IDs our milestone 4 tokenizer produced, with 4 numbers per item:

| Item | Semantic ID |
|---|---|
| Al-Rehab **Jasmin** Perfume Oil, 6 ml | (233, 242, 229, **0**) |
| Al-Rehab **Roses** Perfume Oil, 6 ml | (233, 242, 229, **2**) |
| Al-Rehab **Bakhour** Perfume Oil, 6 ml | (233, 242, 229, **5**) |
| Auric Blends **Black Opium** Perfume Oil | (233, **49**, 184, 0) |
| Cabot Labs **Musk Oil** | (233, **115**, 246, 0) |

Read it like a postal code:

- **First number** `233`: every item here is a perfume oil. That's what this first code came to mean. Nobody told the model "233 = perfume oil"; it emerged from the product text.
- **Second and third numbers**: separate one brand or product line from another. The three Al-Rehab oils share `(233, 242, 229)`.
- **Fourth number**: a tie-breaker. The Al-Rehab oils differ only by scent name, so the first three numbers can't tell them apart (a **collision**). The last number just counts them: 0, 1, 2, … (Part 4 covers collisions.)

### The math: the chain rule of probability

If an item is a tuple of codes $(c_1, c_2, \dots, c_L)$, then "the next item is $i$" means "the next code tuple is $i$'s code tuple". The **chain rule of probability** (true for any random variables, nothing to do with recommenders) splits that into one step per code:

$$
P\big(\text{item} = (c_1, c_2) \mid h\big) = \underbrace{P(c_1 \mid h)}_{\text{step 1: which kind of item?}} \times \underbrace{P(c_2 \mid c_1, h)}_{\text{step 2: which one of that kind?}}
$$

and in general, for $L$ codes:

$$
P\big((c_1, \dots, c_L) \mid h\big) = \prod_{l=1}^{L} P(c_l \mid c_1, \dots, c_{l-1}, h)
$$

The $\prod$ symbol means "multiply all of these together". Each factor is a choice among only $K$ options (in our repo $K = 256$), **not** among all $N$ items.

This is exactly how language models like GPT work: the probability of a sentence is the probability of the first word, times the probability of the second word given the first, and so on. Here the "words" are codes and the "sentence" is one item.

### Worked example

A toy catalog of 6 items with 2-level codes. In reality the codes are learned (Parts 4–5); here we hand-pick them so they're easy to read:

| Item | $c_1$ (kind) | $c_2$ (which one) | Semantic ID |
|---|---|---|---|
| A: shampoo | 0 = hair | 0 | (0, 0) |
| B: conditioner | 0 = hair | 1 | (0, 1) |
| C: hair oil | 0 = hair | 2 | (0, 2) |
| D: red nail polish | 1 = nails | 0 | (1, 0) |
| E: top coat | 1 = nails | 1 | (1, 1) |
| F: perfume | 2 = fragrance | 0 | (2, 0) |

Note that (1, 2), (2, 1) and (2, 2) **are not real items**. Keep that in mind; it comes back in 0.4.

The user's history is $h = (\text{A: shampoo}, \text{D: red polish})$. Suppose our trained model outputs these probabilities.

**Step 1: what kind of item?** $P(c_1 \mid h)$:

| $c_1$ | hair (0) | nails (1) | fragrance (2) |
|---|---|---|---|
| $P(c_1 \mid h)$ | 0.5 | 0.4 | 0.1 |

**Step 2: which one?** The model sees $h$ **and** the code it just picked. Its answer differs for each first code:

| | $c_2 = 0$ | $c_2 = 1$ | $c_2 = 2$ |
|---|---|---|---|
| $P(c_2 \mid c_1 = \text{hair}, h)$ | 0.1 | 0.6 | 0.3 |
| $P(c_2 \mid c_1 = \text{nails}, h)$ | 0.2 | 0.7 | 0.1 |
| $P(c_2 \mid c_1 = \text{fragrance}, h)$ | 0.9 | 0.05 | 0.05 |

**Multiply along each path** (chain rule) to get each item's probability:

| Item | Path | $P(c_1 \mid h) \times P(c_2 \mid c_1, h)$ | Probability | Rank |
|---|---|---|---|---|
| B: conditioner | hair → 1 | $0.5 \times 0.6$ | **0.30** | 1 |
| E: top coat | nails → 1 | $0.4 \times 0.7$ | **0.28** | 2 |
| C: hair oil | hair → 2 | $0.5 \times 0.3$ | 0.15 | 3 |
| F: perfume | fragrance → 0 | $0.1 \times 0.9$ | 0.09 | 4 |
| D: red polish | nails → 0 | $0.4 \times 0.2$ | 0.08 | 5 |
| A: shampoo | hair → 0 | $0.5 \times 0.1$ | 0.05 | 6 |

So the model recommends **conditioner, then top coat**. It's sensible: the user bought shampoo (so conditioner pairs with it) and red polish (so a top coat pairs with it). It also gives low probability to re-buying the same shampoo or polish.

These add up to $0.95$, not $1$. The missing $0.05$ went to code paths that aren't real items: (nails → 2) took $0.4 \times 0.1 = 0.04$, and (fragrance → 1 or 2) took $0.1 \times 0.1 = 0.01$. Remember that leak.

---

## 0.4 Finding the top items quickly: beam search and the trie

In the toy example we computed all 6 items. In our real data that's impossible. With $L = 4$ codes and $K = 256$ values each there are

$$
256^4 \approx 4.3 \text{ billion}
$$

possible code tuples, and only 12,101 of them are real items. We need a shortcut.

### Beam search

**Beam search** keeps only the $B$ most promising partial codes after each step (the **beam width** $B$) and throws the rest away.

Let's redo the toy example with $B = 2$:

| Step | Candidates considered | Keep the top $B = 2$ |
|---|---|---|
| 1 | hair 0.5, nails 0.4, fragrance 0.1 | **hair (0.5)**, **nails (0.4)**. Fragrance is dropped. |
| 2 | (0,0) 0.05, (0,1) 0.30, (0,2) 0.15, (1,0) 0.08, (1,1) 0.28, (1,2) ✗ | **(0,1) = B, 0.30** and **(1,1) = E, 0.28** |

The top 2 come out right: conditioner, top coat. Instead of scoring every combination, we only expanded 2 prefixes.

**The catch:** beam search is a *shortcut*, not an exact search. Perfume F (0.09) has a *higher* probability than red polish D (0.08), but F could never be found with $B = 2$, because its first step (fragrance, 0.1) was already dropped. In practice we use a beam at least as wide as the number of recommendations we want; in our repo $B = 20$ for top-20 lists.

### The trie: never generate a non-existent item

Look at step 2 again: candidate (1, 2) is marked ✗. There's no item with that code. A model that's free to pick any code could happily output (nails → 2), which is a "product" that doesn't exist. This is the generative-model version of **hallucination**.

The fix is a **trie** (prefix tree) of all real Semantic IDs:

```
            (root)
          /   |    \
      hair  nails  fragrance      ← allowed first codes: {0, 1, 2}
     / | \   |  \       |
    0  1  2  0   1      0         ← allowed second codes depend on the first:
    A  B  C  D   E      F            hair → {0,1,2}, nails → {0,1}, fragrance → {0}
```

At every step we look up the prefix generated so far in the trie, and set the probability of every code that would leave the tree to zero. Then (nails → 2) can never be produced. This is called **constrained decoding**, and it guarantees that **every generated code maps to a real item**.

### A practical detail: logs instead of products

With 4 levels, the probabilities being multiplied get tiny. Real code adds **log-probabilities** instead, which is equivalent because $\log(a \times b) = \log a + \log b$, and it avoids numbers too small for a computer to store accurately:

$$
\log P\big((c_1, \dots, c_L) \mid h\big) = \sum_{l=1}^{L} \log P(c_l \mid c_{\lt l}, h)
$$

Ranking by the sum of logs gives exactly the same order as ranking by the product.

---

## 0.5 Why bother? Trade-offs at a glance

Remember the three weaknesses of Approach 1 from 0.2. Here's how Semantic IDs address them, and what they cost:

| | Approach 1: score every item | Approach 2: generate the Semantic ID |
|---|---|---|
| **Item vocabulary** | one vector per item: $N$ vectors (billions at YouTube) | $L \times K$ code tokens: $4 \times 256 = 1{,}024$ in our repo, **no matter how many items** |
| **New items** | random vector until people buy it | gets a meaningful ID on day one, computed from its text |
| **Similar items** | each learns alone | share code prefixes, so they share what the model learned |
| **Fits into an LLM?** | awkward: a separate giant table | natural: codes are just extra "words" in the vocabulary (this is PLUM's main point) |
| **Cost: collisions** | none: every item has its own vector | near-duplicate items can share codes, needing a tie-breaker |
| **Cost: lost detail** | full detail in each vector | squeezing a 768-number embedding into 3 small codes loses detail. In Part 4 this *halves* our kNN accuracy |
| **Cost: inference** | one big matrix multiply | step-by-step beam search, slower per user |

Neither approach wins everywhere. The industrial papers in this repo's reading list are largely about making Approach 2 work well enough to deploy.

---

## 0.6 The map: how this repo's pieces fit together

```
  Amazon reviews                                         Part 1, milestone 1
  (user, item, time)        src/genrec/data/amazon.py
        │
        ▼
  per-user sequences  ──────────────────────────────┐
  h = (i_1, …, i_n)         data/sequences.py      │
        │                                           │
        │   item text: "Title: … Brand: … Price: …" │
        ▼                                           │
  ┌──────────────────────┐                          │
  │ text embedding       │  embeddings/text.py      │   Part 3, milestone 2
  │ 768 numbers per item │                          │
  └──────────────────────┘                          │
        │                                           │
        ▼                                           │
  ┌──────────────────────┐                          │
  │ quantize → Semantic  │  tokenizers/kmeans.py    │   Parts 4–5, milestones 3–4
  │ ID (c1, c2, c3, c4)  │  tokenizers/rqvae.py     │
  └──────────────────────┘                          │
        │                                           │
        ▼                                           ▼
  ┌───────────────────────────────────────────────────┐
  │ generative model: read history as code tokens,   │  models/tiger.py   Part 6, milestone 5
  │ write next item's codes with beam search + trie  │
  └───────────────────────────────────────────────────┘

  Baselines for comparison (Approach 1):            Part 2
    popularity, SASRec          models/baselines.py, models/sasrec.py
    semantic kNN                models/baselines.py   (Part 3)
```

And here is what each stage has measured so far, on the test split. Each later part explains its own row.

| Model | Approach | Test Recall@10 | What it tells us |
|---|---|---|---|
| Popularity | score (no learning) | 0.0114 | the floor |
| Semantic kNN, raw embeddings | score | 0.0593 | product text alone is a strong signal |
| Semantic kNN on Semantic IDs (RQ-VAE) | score | 0.0312 | quantizing to codes throws away about half of that signal |
| SASRec | score (learned) | **0.0753** | the bar to beat |
| TIGER, 2-epoch smoke test only | **generate** | 0.0252 | the pipeline works; not trained properly yet |

Roughly 1 in 13 users' true next item appears in SASRec's top 10, out of 12,101 possible items. That sounds low, but it is a hard task: random guessing would get 10 / 12,101 ≈ 0.0008.

---

## 0.7 Check yourself

**Q1.** In Approach 1, why does the softmax denominator make large catalogs expensive?

It sums $e^{s(h,i')}$ over **every** item in the catalog, so the model has to compute $N$ scores for each user. With billions of items that's both the compute cost and the cost of storing $N$ item vectors.



**Q2.** In the toy example, the user had history (shampoo, red polish). If the model had given $P(c_1 = \text{fragrance} \mid h) = 0.45$ and $P(c_1 = \text{nails} \mid h) = 0.05$ (hair still 0.5), which two items would beam search with $B = 2$ return?

Step 1 keeps hair (0.5) and fragrance (0.45). Step 2 candidates: (0,0) 0.05, (0,1) 0.30, (0,2) 0.15, (2,0) $0.45 \times 0.9 = 0.405$. The trie removes (2,1) and (2,2). Top 2: **F perfume (0.405)** and **B conditioner (0.30)**.



**Q3.** A brand-new lipstick is added to the catalog today. Why can a Semantic-ID model recommend it at all, when SASRec effectively can't?

Its Semantic ID is computed from its **text** (title, brand, category), so it lands near other lipsticks, sharing their first codes. The generative model already knows how to produce those prefixes. SASRec's vector for the new item has never been trained, so its score is essentially random.



**Q4.** Why do we need the 4th "tie-breaker" code?

Because the first three codes are a lossy summary of the product text. Near-duplicates (the six Al-Rehab perfume oils differ only by scent name) can get exactly the same first three codes. Without a 4th code, the model could not tell them apart, and the trie would map one code tuple to six items.



**Q5.** What problem does the trie solve that beam search alone doesn't?

Beam search only decides **which** paths to keep; it can still keep a path that isn't a real item, like (nails → 2). The trie removes those paths at every step, so every output maps to an actual product.



---

## Glossary

| Term | Meaning |
|---|---|
| **Sequential recommendation** | predicting a user's next item from the ordered list of their past items |
| **Top-K list** | the $K$ items a model ranks highest for a user |
| **Embedding** | a list of numbers (vector) representing something, such as an item, a user, or a piece of text |
| **Softmax** | turns any list of scores into probabilities: $e^{s_i} / \sum_j e^{s_j}$ |
| **Cold start** | the difficulty of recommending items (or users) with no interaction history |
| **Semantic ID** | a short tuple of small integers per item, where similar items share prefixes |
| **Codebook** | the $K$ possible values at one level of a Semantic ID (and, in Parts 4–5, the vector each value stands for) |
| **Collision** | two different items receiving the same code tuple |
| **Chain rule (probability)** | $P(a, b) = P(a) \times P(b \mid a)$, extended to any number of steps |
| **Beam search** | step-by-step decoding that keeps only the $B$ best partial sequences at each step |
| **Trie** | a prefix tree; here, all valid Semantic IDs, used to block invalid codes during decoding |
| **Generative retrieval** | retrieving items by generating their identifiers instead of scoring the whole catalog |

**Next:** Part 1, Data and evaluation: where these sequences come from, and how exactly we score a top-K list.
