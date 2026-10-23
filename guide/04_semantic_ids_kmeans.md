# Part 4: Semantic IDs with k-means

> **After this part you should be able to explain:** how k-means turns vectors into group numbers, how **residual quantization** stacks several k-means steps to build a multi-level code, how that differs from a hierarchical tree, what collisions and codebook usage are, and why quantization costs retrieval quality.

This is where items get their Semantic IDs: from Part 3's 768 numbers per item down to 3 small integers.

---

## 4.1 Intuition: quantization is rounding

**Quantization** means replacing a value with the nearest one from a small fixed menu. You do it every time you round a price to the nearest dollar.

A familiar example: old GIF images could only use **256 colours**. Each pixel's true colour, one of 16 million, was replaced by the nearest colour in a 256-colour **palette**, and the image stored only the palette index (0–255). The picture got slightly worse, and much smaller.

Semantic IDs do the same for items:

- the "true colour" is the item's 768-number embedding $`\mathbf{e}_i`$,
- the "palette" is a **codebook** of $`K`$ representative vectors,
- the stored "index" is the item's **code**.

One code from a palette of 256 is too coarse to tell 12,101 items apart. So we'll use **several codes per item**, each one refining the last.

---

## 4.2 k-means: learning the palette

### The math

Given points $`\mathbf{x}_1, \dots, \mathbf{x}_n`$, k-means looks for $`K`$ **centroids** $`\mu_1, \dots, \mu_K`$ (the codebook) and an assignment $`c(j) \in \lbrace 0, \dots, K-1\rbrace `$ of each point to one centroid. The goal is to make each point as close as possible to its centroid:

$$
\min_{\mu,~ c} \quad  \sum_{j=1}^{n} \big\Vert \mathbf{x}_j - \mu_{c(j)}\big\Vert ^2
$$

That's the total squared distance, also called the **quantization error**. The standard algorithm (**Lloyd's algorithm**) alternates two easy steps until nothing changes:

1. **Assign:** send each point to its nearest centroid, $`c(j) = \arg\min_k \Vert \mathbf{x}_j - \mu_k\Vert ^2`$.
2. **Update:** move each centroid to the average of its points, $`\mu_k = \text{mean}\lbrace \mathbf{x}_j : c(j) = k\rbrace `$.

Each step can only lower the error, so it always settles. It may settle in a *local* optimum, which depends on the starting centroids.

### Worked example (1-D, $`K = 2`$)

Points: $`1, 2, 3, 10, 11, 12`$. Deliberately bad start: $`\mu_0 = 1`$, $`\mu_1 = 2`$.

| Round | Centroids $`(\mu_0, \mu_1)`$ | Assign | Update |
|---|---|---|---|
| 1 | (1, 2) | 1 → 0; everything else is closer to 2 → 1 | $`\mu_0 = 1`$, $`\mu_1 = (2+3+10+11+12)/5 = 7.6`$ |
| 2 | (1, 7.6) | 1, 2, 3 → 0 (e.g. 3 is 2 from 1 and 4.6 from 7.6); 10, 11, 12 → 1 | $`\mu_0 = 2`$, $`\mu_1 = 11`$ |
| 3 | (2, 11) | same as before | no change: **done** |

Final codebook $`\lbrace 2, 11\rbrace `$. Points 1–3 get code 0, points 10–12 get code 1. Each point is now stored as one bit, at the cost of being "rounded" to 2 or 11.

---

## 4.3 Residual quantization: codes that refine each other

### Intuition

After one round of k-means, each item is "rounded" to its centroid, and there's a leftover error, the **residual**. Residual quantization runs k-means **again on the leftovers**, then again on what's left after that. Each level corrects the previous level's rounding error, like describing a location by *country → city → street*.

### The math

Start with $`\mathbf{r}_0 = \mathbf{x}`$ (the item's embedding). For each level $`l = 1, \dots, L`$, with its own codebook $`\lbrace \mu^{(l)}_k\rbrace `$:

$$
c_l = \arg\min_k \big\Vert \mathbf{r}_{l-1} - \mu^{(l)}_k\big\Vert , \qquad \mathbf{r}_l = \mathbf{r}_{l-1} - \mu^{(l)}_{c_l}
$$

The Semantic ID is $`(c_1, \dots, c_L)`$, and the item's **reconstruction** (its quantized vector) is the sum of the chosen centroids:

$$
\hat{\mathbf{x}} = \mu^{(1)}_{c_1} + \mu^{(2)}_{c_2} + \dots + \mu^{(L)}_{c_L}
$$

Each level's codebook is learned by running k-means on the residuals of **all** items at that level. That means code 5 at level 2 means the same *correction* no matter what the level-1 code was. OneRec reports using this method under the name **RQ-KMeans**.

### Worked example (2-D, 3 levels, $`K = 4`$)

Item $`\mathbf{x} = (3.2, 0.9)`$, with codebooks that get finer at each level:

| Level | Codebook (codes 0, 1, 2, 3) | Residual in | Nearest code | Residual out |
|---|---|---|---|---|
| 1 | (0,0), **(3,0)**, (0,3), (3,3) | (3.2, 0.9) | **1** → (3, 0), squared distance 0.85 | (0.2, 0.9) |
| 2 | (0,0), (1,0), **(0,1)**, (1,1) | (0.2, 0.9) | **2** → (0, 1), squared distance 0.05 | (0.2, −0.1) |
| 3 | (0,0), **(0.25,0)**, (0,0.25), (0.25,0.25) | (0.2, −0.1) | **1** → (0.25, 0), squared distance 0.0125 | (−0.05, −0.1) |

- **Semantic ID:** $`(1, 2, 1)`$
- **Reconstruction:** $`(3, 0) + (0, 1) + (0.25, 0) = (3.25, 1.0)`$, versus the true $`(3.2, 0.9)`$
- The error shrank at every level: squared distance 0.85, then 0.05, then 0.0125.

### How many IDs are possible?

With $`L`$ levels of $`K`$ codes there are $`K^L`$ possible IDs. With $`K = 256`$ and $`L = 3`$ that's $`256^3 \approx 16.8`$ million, far more than 12,101 items. So in principle every item could get its own ID. In practice similar items get pulled to the same codes, which brings us to collisions.

---

## 4.4 The alternative: hierarchical k-means (a tree)

Instead of quantizing residuals, **hierarchical k-means** builds a tree. It clusters all items into $`K`$ groups, then runs a separate k-means **inside each group**, and so on.

| | Residual k-means | Hierarchical k-means |
|---|---|---|
| Level-2 codebook | one, shared by all items | a separate one inside each level-1 group |
| Meaning of "code 5 at level 2" | the same correction everywhere | something different under every parent |
| Small groups | still quantized | a group with $`\le K`$ items just numbers its members 0, 1, 2, … |

That last row matters. Once a group has $`K`$ or fewer items, there's nothing left to cluster. Our implementation (like k-means with $`K \ge n`$) gives each item its own code, which is a label with no meaning. With $`K = 256`$, level-1 groups average about 47 items, so **level 2 already numbers individual items and level 3 is unused**. The tree has zero collisions, but only by construction.

---

## 4.5 Collisions and the tie-breaker code

A **collision** is two or more items with exactly the same code tuple. With residual k-means $`3 \times 256`$, 11.4% of our items share their ID with at least one other item. The largest groups, from notebook 03:

| Shared ID | Items |
|---|---|
| (66, 116, 94) | six Red Cherry false-eyelash packs: #82, #138, #43, #WSP, #117, and #43 in a 12-pack |
| (73, 5, 5) | six China Glaze nail polishes: Nova, Westside Warrior, Exotic Encounters, … |

These are **near-duplicates**: their texts differ only by a shade name or number, so their embeddings are almost identical. No quantizer could separate them better than the text does.

**The fix (from TIGER):** add a 4th code that numbers the items within each collision group: 0, 1, 2, …. Then every item has a unique ID, like (73, 5, 5, **0**), (73, 5, 5, **1**), …. The 4th code carries no meaning; it's just a tie-breaker.

---

## 4.6 Measuring a tokenizer

How do we judge a set of Semantic IDs without training a recommender on them? Four measurements:

### 1. Collision rate

The share of items whose ID is shared with another item. Lower is better.

### 2. Codebook usage (perplexity)

If a level has 256 codes but almost every item uses the same 5, that level carries little information. **Perplexity** measures the *effective* number of codes in use. If code $`k`$ is used by a fraction $`p_k`$ of items:

$$
\text{perplexity} = \exp\Big(-\sum_k p_k \log p_k\Big)
$$

The expression in brackets is the **entropy**. Examples with 4 codes:

| Usage $`(p_0, p_1, p_2, p_3)`$ | Perplexity | Reading |
|---|---|---|
| (0.25, 0.25, 0.25, 0.25) | 4.00 | all four codes fully used |
| (0.97, 0.01, 0.01, 0.01) | 1.18 | effectively one code |

### 3. Category purity

For all items sharing a prefix (say, the same level-1 code), what share belong to the group's most common leaf category? High purity means the codes really mean something like "product type".

### 4. Retrieval signal kept

Run Part 3's semantic kNN on the **reconstructions** $`\hat{\mathbf{x}}`$ instead of the original embeddings. The drop in score shows how much useful detail quantization threw away.

---

## 4.7 Our results

### Codebook size (3 levels)

| Tokenizer | Unique IDs | Collision rate | Perplexity, levels 1 / 2 / 3 | kNN valid NDCG@10 |
|---|---|---|---|---|
| Residual, $`K = 32`$ | 7,278 | 61.1% | 30 / 28 / 28 | 0.0089 |
| Residual, $`K = 64`$ | 9,697 | 33.2% | 57 / 53 / 49 | 0.0129 |
| Residual, $`K = 128`$ | 10,975 | 16.7% | 115 / 97 / 88 | 0.0147 |
| **Residual, $`K = 256`$** | **11,354** | **11.4%** | **228 / 127 / 49** | **0.0157** |
| Hierarchical, $`K = 32`$ | 11,971 | 1.7% | 30 / 32 / 22 | 0.0306 |
| Hierarchical, $`K = 256`$ | 12,101 | 0% | 228 / 76 / **1** | 0.0345 |
| *(raw embeddings, no quantization)* | | | | *0.0346* |

- A bigger $`K`$ means fewer collisions and better kNN.
- At $`K = 256`$ the residual tokenizer's **3rd level is underused** (perplexity 49 of 256): the leftovers at that depth are small.
- Hierarchical looks almost perfect, but its last levels just number items (perplexity → 1). Its reconstruction is nearly the item itself, so it hasn't really compressed anything. We carry **residual 3×256** forward.

### What prefixes mean

Category purity (the largest single category is 6% of all items, so random grouping would give about 6%):

| Tokenizer | 1-code prefix | 2-code prefix | 3-code prefix |
|---|---|---|---|
| Residual 3×256 | 57% | 84% | 98% |

Two real level-1 groups: code 189 is 60 items, mostly *Eau de Toilette* (46 of 60). Code 3 is 108 items of sunscreens and SPF lotions.

### The cost: quantization halves retrieval quality

| Test split | Recall@10 | NDCG@10 |
|---|---|---|
| Semantic kNN, raw embeddings | 0.0593 | 0.0350 |
| Semantic kNN on residual 3×256 reconstructions | **0.0302** | **0.0156** |

Why so large a drop, when the average reconstruction cosine is a high-sounding **0.970**? Compare it with how close an item's real neighbours are (median cosines, from notebook 03):

| Item's … | Median cosine |
|---|---|
| nearest neighbour | 0.955 |
| 10th nearest neighbour | 0.928 |
| own reconstruction $`\hat{\mathbf{x}}`$ | 0.970 |

**The rounding error is about as big as the distance to the item's nearest neighbour.** Once quantized, an item could just as well be any of its close neighbours, so fine-grained ranking is gone. This is the anisotropy from Part 3: in a narrow cone, "0.97" isn't close.

That's the central trade-off of Semantic IDs. They're a **coarse, shared vocabulary**: great for grouping, poor for fine ranking. In Part 6 the sequence model has to supply the fine ranking itself.

### A reproducibility caveat

Two identical residual 3×256 runs gave 11,354 vs 11,447 unique IDs, and kNN NDCG@10 of 0.0157 vs 0.0172. k-means here isn't bit-for-bit repeatable (probably because of multithreaded arithmetic). Don't read much into differences of that size.

### In our code

- [kmeans.py](../src/genrec/tokenizers/kmeans.py):
  - `residual_kmeans` and `hierarchical_kmeans` return `(codes, recon)`.
  - `add_collision_suffix` adds the 4th code.
  - `code_stats` computes collisions, usage and perplexity.
  - `quantized_embeddings` normalizes reconstructions for kNN.
- [build_semantic_ids.py](../scripts/build_semantic_ids.py): `--method residual --k 256` writes `data/processed/beauty/semantic_ids/residual_kmeans-3x256.npy`.
- [Notebook 03](../notebooks/03_semantic_ids_kmeans.ipynb): all the tables above.

---

## 4.8 Check yourself

<details>
<summary><b>Q1.</b> In the 2-D residual example, what ID and reconstruction would <b><i>x</i></b> = (0.1, 3.3) get?</summary>

Level 1: nearest to (0.1, 3.3) is (0, 3), code 2; residual (0.1, 0.3). Level 2: nearest to (0.1, 0.3) is (0, 0), code 0 (squared distance 0.10, against 0.50 for (0, 1)); residual (0.1, 0.3). Level 3: nearest is (0, 0.25), code 2 (squared distance 0.0125); residual (0.1, 0.05). ID **(2, 0, 2)**, reconstruction **(0, 3.25)**.
</details>

<details>
<summary><b>Q2.</b> A level uses all 256 codes, but with perplexity 49. What does that mean?</summary>

Every code is used at least once, but the usage is very uneven: the level carries about as much information as if only 49 codes were used equally. A few codes take most of the items.
</details>

<details>
<summary><b>Q3.</b> Hierarchical k-means with <i>K</i> = 256 has zero collisions. Why isn't it the better tokenizer?</summary>

Because it gets there by giving up on meaning. Its level-1 groups average about 47 items, which is fewer than 256, so level 2 simply numbers the items inside each group, and level 3 is always 0. Those codes are arbitrary labels that mean different things under each parent, so a generative model can't learn anything general about them.
</details>

<details>
<summary><b>Q4.</b> Why can't any tokenizer separate the six China Glaze polishes?</summary>

Their texts differ only by the shade name (e.g. "Nova" vs "Street Chic"), so their embeddings are nearly identical. A tokenizer only sees the embeddings. The 4th tie-breaker code is how they get unique IDs.
</details>

---

**Next:** [Part 5, RQ-VAE](05_rqvae.md): learning a better space to quantize in.
