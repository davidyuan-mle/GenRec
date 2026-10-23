# Part 3: Text Embeddings and Semantic kNN

> **After this part you should be able to explain:** how a product's text becomes a vector of 768 numbers, why cosine similarity measures "how alike" two products are, how a simple recommender can work from text similarity alone, and what our experiments say about content vs. purchase behaviour.

This is the first step toward Semantic IDs: before we can give items meaningful *codes*, we need meaningful *vectors*.

---

## 3.1 Two kinds of knowledge about an item

SASRec learns what an item is **only from who bought it**. Its vector $\mathbf{E}[i]$ starts random and is shaped by co-purchases. That's called **collaborative** information.

But every product also comes with a **description**. We know, without any purchase data, that "Covergirl LashBlast Mascara, Very Black" is a mascara and that it's similar to "Covergirl LashExact Mascara". That's **content** information.

| | Collaborative (SASRec) | Content (this part) |
|---|---|---|
| Learned from | who bought what, in which order | the item's own text |
| Knows a brand-new item? | no | yes, from its description |
| Knows "people who buy X also buy Y" (complements)? | yes | no |
| Knows near-duplicates are similar? | only with enough purchases | yes, immediately |

Semantic IDs are built from content, which is where their cold-start advantage comes from.

---

## 3.2 From text to a vector

### Step 1: write one string per item

We build one sentence per item from its metadata, the same recipe TIGER uses (title, brand, categories, price). A real example from our data:

```
Title: Suave Naturals Gentle Cleansing Shampoo, Waterfall Mist, 12 oz. Brand: Suave.
Categories: Beauty, Hair Care, Shampoos. Price: $5.95
```

Missing fields are skipped. The texts average about 24 words. The long free-text description is left out on purpose: it's noisy, and these four fields carry most of the identity.

### Step 2: run it through a sentence encoder

A **sentence encoder** is a Transformer (Part 2) pre-trained by someone else on huge amounts of text, so that **sentences with similar meaning get similar vectors**. You don't train it; you just use it.

$$
\text{text}_i \;\xrightarrow{\;\text{encoder}\;}\; \mathbf{e}_i \in \mathbb{R}^{768}
$$

We use `sentence-t5-base`, a smaller version of the Sentence-T5 encoder the TIGER paper used. Encoding all 12,101 items took under 2 minutes on a laptop GPU.

Then we **normalize** each vector to length 1:

$$
\mathbf{e}_i \leftarrow \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|}, \qquad \text{where } \|\mathbf{e}\| = \sqrt{e_1^2 + \dots + e_{768}^2}
$$

---

## 3.3 Measuring similarity: cosine

### The math

The **cosine similarity** of two vectors measures the angle between them and ignores their lengths:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

It's 1 when they point the same way, 0 when they're at right angles (unrelated), and −1 when they're opposite. **When both vectors have length 1, cosine is just the dot product**, which is why we normalize: everything afterwards is a cheap dot product.

### Worked example (2 dimensions)

| Vector | Value | Length |
|---|---|---|
| mascara A | $(0.8, 0.6)$ | 1 |
| mascara B | $(0.6, 0.8)$ | 1 |
| shampoo | $(0.6, -0.8)$ | 1 |

- $\cos(\text{mascara A}, \text{mascara B}) = 0.8 \cdot 0.6 + 0.6 \cdot 0.8 = 0.96$: very similar.
- $\cos(\text{mascara A}, \text{shampoo}) = 0.8 \cdot 0.6 + 0.6 \cdot (-0.8) = 0$: unrelated.

### Does it work on our data?

Yes. Real nearest neighbours by cosine, from notebook 02:

| Query item | Nearest neighbours (cosine) |
|---|---|
| Covergirl Clump Crusher Mascara, Very Black | Covergirl LashExact Mascara (0.975), Covergirl LashBlast 24 Hour Mascara (0.960), Covergirl LashBlast Fusion (0.959), … |
| All Eyes on You Eyelash & Brow Serum | RapidLash Eyelash and Eyebrow Serum (0.950), SmartLash Eyelash Enhancer (0.950), … |

And at scale: **52%** of each item's 10 nearest neighbours share its leaf category (e.g. "Mascara"). For 10 random items it's only **2%**.

### A quirk: all the numbers are high

Two *random* beauty products already have a median cosine of **0.84**. Consecutive purchases by the same user have a median of **0.864**. That looks like a tiny gap, but it isn't a problem for ranking: what matters is that the true neighbours rank above the rest, and they do. Think of it as every product living in a fairly narrow cone, with differences that are small in absolute terms but consistent.

This sentence-encoder trait is called **anisotropy**. Remember it: in Part 4 it explains why "a 0.97 match" is not as close as it sounds.

---

## 3.4 Semantic kNN: recommending by similarity

### Intuition

The simplest content-based recommender: "recommend the items whose text is most similar to what you bought recently". **kNN** stands for k-nearest neighbours. No training is needed.

### The math

1. Build a **user profile** from the last $m$ items, with more recent items counting more. Item $a$ steps back from the most recent gets weight $\gamma^a$ (gamma to the power $a$), with **decay** $0 < \gamma \le 1$:

$$
\mathbf{p} = \sum_{a=0}^{m-1} \gamma^{a} \, \mathbf{e}_{i_{n-a}} = \mathbf{e}_{i_n} + \gamma\, \mathbf{e}_{i_{n-1}} + \gamma^2\, \mathbf{e}_{i_{n-2}} + \dots
$$

2. Score every item: $s(i) = \mathbf{p} \cdot \mathbf{e}_i$.
3. **Exclude items already in the history.** Otherwise the most recent item, with similarity ≈ 1 to itself, would always come first.
4. Return the top K.

With $\gamma = 1$ all recent items count equally. With small $\gamma$ the most recent item dominates.

### Worked example

Five items with 2-D vectors. This is the exact example in the repo's unit test:

| Item | $\mathbf{e}_i$ |
|---|---|
| 1 | $(1, 0)$ |
| 2 | $(1, 0)$ |
| 3 | $(0, 1)$ |
| 4 | $(0.6, -0.8)$ |
| 5 | $(0, 1)$ |

History $= (3, 1)$, so item 1 is most recent. With $m = 2$ and $\gamma = 0.5$:

$$
\mathbf{p} = 1 \cdot \mathbf{e}_1 + 0.5 \cdot \mathbf{e}_3 = (1, 0) + (0, 0.5) = (1, 0.5)
$$

| Item | $\mathbf{p} \cdot \mathbf{e}_i$ | Note |
|---|---|---|
| 1 | 1.0 | in history: excluded |
| 2 | **1.0** | rank 1 |
| 3 | 0.5 | in history: excluded |
| 5 | **0.5** | rank 2 |
| 4 | $0.6 - 0.4 = 0.2$ | rank 3 |

Top 2: **(2, 5)**. If the history order flips to $(1, 3)$, then $\mathbf{p} = (0.5, 1)$ and the top 2 becomes **(5, 2)**. The most recent item decides the order.

---

## 3.5 Results

### Tuning the decay on validation

From notebook 02 (validation split, so the test stays untouched):

| Profile | Valid NDCG@10 |
|---|---|
| last item only ($m = 1$) | 0.0350 |
| last 5, $\gamma = 0.3$ (best) | 0.0359 |
| last 5, $\gamma = 0.5$ (our default) | 0.0346 |
| last 5, $\gamma = 1.0$ (plain average) | 0.0234 |
| last 10, $\gamma = 1.0$ | 0.0217 |

**Recency dominates:** what you bought last is far more informative than your history as a whole. A plain average is clearly worst. The best setting barely beats the default, so we kept the default.

### Test results

| Test split | Recall@10 | NDCG@10 | Recall@20 | Training? |
|---|---|---|---|---|
| Popularity | 0.0114 | 0.0053 | 0.0195 | none |
| **Semantic kNN** | **0.0593** | **0.0350** | 0.0790 | **none** |
| SASRec | 0.0753 | 0.0423 | 0.1077 | 50 epochs |

Three lessons:

1. **Content alone gets far.** With zero training, kNN reaches about 80% of SASRec at K = 10.
2. **SASRec's lead grows at K = 20** (0.108 vs 0.079). kNN only finds "more of the same". SASRec also learns **complements**, like shampoo → conditioner, which look different in text but go together in purchases.
3. **kNN has no validation-to-test drop** (Recall@10 0.0594 → 0.0593), while popularity and SASRec both drop. A plausible reading: purchase patterns shift over time, and product descriptions don't.

### In our code

- [text.py](../src/genrec/embeddings/text.py): `encode_texts` and `item_embeddings`. The latter adds a zero row 0 for padding, so row $i$ is item $i$.
- [embed_items.py](../scripts/embed_items.py): saves the matrix to `data/processed/beauty/embeddings/sentence-t5-base.npy`, shape $12{,}102 \times 768$.
- [amazon.py](../src/genrec/data/amazon.py): `item_text` builds the string.
- [baselines.py](../src/genrec/models/baselines.py): `SemanticKNN`. [semantic_knn_beauty.yaml](../configs/semantic_knn_beauty.yaml) sets $m = 5$, $\gamma = 0.5$.
- [Notebook 02](../notebooks/02_semantic_embeddings.ipynb): neighbours, category check, similarity histogram, sweep.

---

## 3.6 Check yourself

<details>
<summary><b>Q1.</b> Why do we normalize the embeddings to length 1?</summary>

So cosine similarity equals the plain dot product. That makes scoring every item a single matrix multiply, and it stops long vectors from scoring high just because they're long.
</details>

<details>
<summary><b>Q2.</b> With $\gamma = 0.5$ and $m = 3$, what weights do the last three items get?</summary>

Most recent: $0.5^0 = 1$. One back: $0.5$. Two back: $0.25$.
</details>

<details>
<summary><b>Q3.</b> Why must kNN exclude history items, when SASRec doesn't need to?</summary>

The most recent item has cosine ≈ 1 with itself, so it would always be recommended first, and the near-duplicates of other history items would follow. SASRec *learns* from data that people rarely re-buy the exact same item, so it naturally gives low scores to what you just bought.
</details>

<details>
<summary><b>Q4.</b> A user buys shampoo. Which model is more likely to recommend the matching conditioner, kNN or SASRec, and why?</summary>

SASRec. Shampoo and conditioner texts are only moderately similar, so kNN prefers other shampoos. SASRec has seen many sequences where conditioner follows shampoo, which is a collaborative signal that text similarity can't see.
</details>

---

**Next:** [Part 4, Semantic IDs with k-means](04_semantic_ids_kmeans.md): compressing each 768-number vector into a few integers.
