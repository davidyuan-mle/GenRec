# Part 2: Baselines (Popularity and SASRec)

> **After this part you should be able to explain:** why every experiment needs a "dumb" baseline, how self-attention lets a model look back over a user's history, how SASRec is trained to predict the next item at every position at once, and what its training curve tells us.

These are both **Approach 1** models from Part 0: they score every item in the catalog.

---

## 2.1 Popularity: the floor

### Intuition

Recommend the same thing to everyone: the items that were bought most often. It ignores the user completely. If a fancy model can't beat this, something is wrong with the model (or the evaluation).

### The math

Count how often each item appears in the training sequences:

$$
\text{count}(i) = \text{number of times item } i \text{ appears in all users' training items}
$$

and recommend the top-K items by count to every user.

### Worked example

Training sequences: (A, B, C), (A, C), (B, A, D). Counts: A = 3, B = 2, C = 2, D = 1. Every user gets the top-2 list (A, B). (B and C are tied; the code takes whichever it counted first.)

### In our code and results

[baselines.py](../src/genrec/models/baselines.py), class `Popularity`, is about 10 lines.

| Test split | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| Popularity | 0.0114 | 0.0053 | 0.0040 |

That's about 14× random guessing (0.0008), thanks to the popularity skew from Part 1: the top 10% of items cover 43% of interactions. Every model after this must clearly beat 0.0114.

---

## 2.2 SASRec: the idea

**SASRec** (Self-Attentive Sequential Recommendation, Kang & McAuley 2018) applies the Transformer, the architecture behind GPT, to item sequences:

- Each **item** is treated like a **word**.
- A user's **history** is treated like a **sentence**.
- The model predicts the **next item** the same way GPT predicts the next word.

To do that, it needs three ingredients, which the next sections cover one at a time:

1. **Embeddings**: turn item IDs and positions into vectors (2.3).
2. **Self-attention**: let each position look back at earlier positions (2.4).
3. **Output scores and the loss**: score every item, and learn from mistakes (2.5).

---

## 2.3 Ingredient 1: item and position embeddings

### The math

- An **item embedding table** $\mathbf{E} \in \mathbb{R}^{(N+1) \times d}$: row $i$ is item $i$'s vector. Row 0 is reserved for **padding** (explained below). In our config $d = 64$.
- A **position embedding table** $\mathbf{P} \in \mathbb{R}^{L \times d}$: one vector per position in the sequence. Our maximum length is $L = 50$.

The input vector at position $t$ is the sum of the two:

$$
\mathbf{x}_t = \mathbf{E}[i_t] + \mathbf{P}[t]
$$

Why add a position vector? Attention (next section) treats its inputs like an unordered bag. Adding $\mathbf{P}[t]$ is how the model knows that "shampoo, then conditioner" differs from "conditioner, then shampoo".

### Padding

A batch needs every sequence to have the same length, but users have different history lengths. So shorter histories are filled with the special item 0 on the **left**:

```
max length 6, history (A, B, C)  →  (0, 0, 0, A, B, C)
```

Left padding keeps the most recent item at the same position (the last one) for every user. Padding positions are masked out so they never influence real positions. The repo has a test for exactly this: `test_sasrec_ignores_left_padding` in [test_data.py](../tests/test_data.py).

---

## 2.4 Ingredient 2: self-attention

### Intuition

To predict what comes after position $t$, the model should look back at the earlier items and decide **which ones matter**. If the history is (shampoo, nail polish, conditioner), the hair items are more informative for a hair-care prediction than the polish. Attention computes those "how much does each earlier item matter" weights from the data.

### The math

Stack the input vectors into a matrix $\mathbf{X}$ (one row per position). Three learned matrices turn each row into three roles:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad (\text{queries}), \qquad \mathbf{K} = \mathbf{X}\mathbf{W}_K \quad (\text{keys}), \qquad \mathbf{V} = \mathbf{X}\mathbf{W}_V \quad (\text{values})
$$

Think of a library search. The **query** is what position $t$ is looking for. Each **key** describes what an earlier position offers. The **value** is the content that position hands over if chosen.

The attention output is:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}
$$

Step by step:

1. $\mathbf{Q}\mathbf{K}^\top$: entry $(t, s)$ is the dot product of position $t$'s query with position $s$'s key, meaning "how relevant is $s$ to $t$".
2. $\div \sqrt{d}$: keeps the numbers from getting too large as $d$ grows, which would make the softmax too peaked.
3. $+\mathbf{M}$: the **causal mask**. It is $-\infty$ wherever $s > t$ (the future) or $s$ is padding, and 0 elsewhere. Since $e^{-\infty} = 0$, those positions get exactly zero weight. Without it, the model could "cheat" during training by looking at the answer.
4. $\text{softmax}$ along each row: turns relevance scores into weights that are positive and add up to 1.
5. $\times~ \mathbf{V}$: each position's output is a **weighted average** of the values it's allowed to see.

### Worked example

Three positions with $d = 2$. To keep the arithmetic short, take $\mathbf{W}_Q = \mathbf{W}_K = \mathbf{W}_V = $ identity, so $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$:

| Position | Item | $\mathbf{x}_t$ |
|---|---|---|
| 1 | shampoo | $(1, 0)$ |
| 2 | nail polish | $(0, 1)$ |
| 3 | conditioner | $(1, 1)$ |

**Position 3** may look at positions 1, 2 and 3. Its query is $(1, 1)$:

| Looks at | $\mathbf{q}_3 \cdot \mathbf{k}_s$ | $\div \sqrt{2}$ | $e^{(\cdot)}$ | weight |
|---|---|---|---|---|
| 1: shampoo | $1$ | $0.707$ | $2.03$ | $2.03 / 8.17 = 0.248$ |
| 2: polish | $1$ | $0.707$ | $2.03$ | $0.248$ |
| 3: conditioner | $2$ | $1.414$ | $4.11$ | $0.503$ |

Output: $0.248 \cdot (1,0) + 0.248 \cdot (0,1) + 0.503 \cdot (1,1) = (0.75, 0.75)$.

**Position 2** may only look at positions 1 and 2. The mask hides position 3. Its query is $(0, 1)$, so the scores are $0$ and $1$, which become weights $0.33$ and $0.67$. The output is $(0.33, 0.67)$. Position 2 never sees the conditioner, which is exactly what we want when training it to *predict* the conditioner.

### The rest of a Transformer layer

Real layers wrap attention with a few standard parts. You don't need their details to follow this guide:

- **Multiple heads:** run several attentions in parallel, each with its own $\mathbf{W}$ matrices, so different heads can track different patterns (brand, category, price). We use 2 heads.
- **Feed-forward network:** a small 2-layer neural network applied to each position separately after attention.
- **Residual connections and layer normalization:** add each sub-layer's input back to its output and rescale. This keeps training stable.

We stack **2** such layers. The final output at position $t$ is a vector $\mathbf{h}_t \in \mathbb{R}^{64}$ summarizing "the history up to $t$".

---

## 2.5 Ingredient 3: scoring every item, and the loss

### The math

Score every item by its dot product with $\mathbf{h}_t$, reusing the same item table $\mathbf{E}$ as the input (called **weight tying**):

$$
s_t(i) = \mathbf{h}_t \cdot \mathbf{E}[i], \qquad P(i_{t+1} = i \mid i_1, \dots, i_t) = \text{softmax}_i\big(s_t(i)\big)
$$

This is exactly Part 0's Approach 1, with $\mathbf{u}_h = \mathbf{h}_t$ and $\mathbf{v}_i = \mathbf{E}[i]$.

Training minimizes the **cross-entropy loss**: the negative log-probability the model gave to the true next item.

$$
\mathcal{L} = -\log P(i_{t+1} = \text{true item} \mid i_1, \dots, i_t)
$$

| Model's probability for the true item | Loss $-\log P$ |
|---|---|
| 0.77 (confident and right) | 0.26 |
| 0.06 (thought it unlikely) | 2.81 |
| $1/12{,}101$ (no idea: uniform guess) | 9.40 |

Lower is better. The loss explodes as the probability of the truth goes to 0, so the model is pushed hard to never be confidently wrong.

### A useful trick: predict at every position at once

Because of the causal mask, position $t$ only sees items $1 \dots t$. So one pass over a sequence gives a prediction at **every** position, and each can be trained against its own next item:

```
input  :  0    0   A   B   C        (left-padded history)
target :  0    0   B   C   D        (the same sequence shifted by one)
          ↑    ↑   predict B from (A); C from (A,B); D from (A,B,C)
       ignored (padding)
```

One user with $n$ training items gives $n - 1$ training signals in a single forward pass. Part 6 shows that TIGER can't use this trick as easily.

### Interpreting the loss value

$e^{\text{loss}}$ is the **perplexity**: roughly "the model is as uncertain as if choosing uniformly among this many items".

| | Loss | $e^{\text{loss}}$ |
|---|---|---|
| Uniform guess over the catalog | 9.40 | 12,101 items |
| SASRec after 5 epochs | 7.96 | ~2,900 items |
| SASRec after 50 epochs | 6.00 | ~400 items |

The model went from "no idea" to "it's one of about 400 items".

---

## 2.6 Training SASRec in practice

### Early stopping

Train in **epochs** (one pass over all training sequences). Every 5 epochs, compute validation NDCG@10. Keep a copy of the weights from the best validation score. If 4 checks in a row don't improve, stop. At the end, load the best weights and evaluate once on test.

### Our run (from milestone 1)

| Epoch | Train loss | Valid Recall@10 | Valid NDCG@10 |
|---|---|---|---|
| 5 | 7.96 | 0.0464 | 0.0229 |
| 10 | 7.41 | 0.0706 | 0.0369 |
| 20 | 6.80 | 0.0887 | 0.0465 |
| 30 | 6.45 | 0.0966 | 0.0517 |
| 40 | 6.20 | 0.0951 | 0.0517 |
| 50 | 6.00 | 0.0950 | 0.0524 |

From epoch 30 onwards the **training loss keeps falling but validation stays flat**. That's the start of **overfitting**: the model is getting better at the training sequences without getting better at new predictions. That's why we capped training at 50 epochs.

### Final result

| Test split | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| Popularity | 0.0114 | 0.0053 | 0.0040 |
| **SASRec** | **0.0753** | **0.0423** | **0.0346** |

About 6.6× popularity. Validation was higher (Recall@10 0.0950). Popularity shows a similar validation-to-test drop, which suggests the test targets are simply harder; see the leave-one-out subtlety in Part 1.

### In our code

- [sasrec.py](../src/genrec/models/sasrec.py): the model. `encode` does embeddings plus masked attention, `logits` scores every item, `loss` is the cross-entropy, and `recommend` returns the top-K from the last position.
- [sequences.py](../src/genrec/data/sequences.py): `NextItemDataset` builds the shifted input/target pairs. `left_pad` does the padding.
- [train.py](../scripts/train.py): `fit_with_early_stopping` is the loop above (shared with TIGER in Part 6). [sasrec_beauty.yaml](../configs/sasrec_beauty.yaml) has the settings.

### SASRec's weaknesses, revisited

SASRec is the bar the generative model has to beat, and it's strong. But it has exactly the Part 0 weaknesses: one learned vector per item (12,101 × 64 here), no knowledge at all about items it hasn't seen in training (like the 33 test targets from Part 1), and no sharing between near-identical products. Parts 3–6 attack those weaknesses.

---

## 2.7 Check yourself

<details>
<summary><b>Q1.</b> What would go wrong if we removed the causal mask during training?</summary>

Position $t$ could attend to position $t+1$, which is the very item it's supposed to predict. The model would learn to copy the answer, getting near-zero training loss, and would be useless at test time, where the next item isn't in the input.
</details>

<details>
<summary><b>Q2.</b> In the attention example, position 3's weights were (0.248, 0.248, 0.503). Why does the conditioner get the most weight?</summary>

Its key $(1, 1)$ has the largest dot product with the query $(1, 1)$: $2$, against $1$ for the other two. With identity projections, a position attends most to positions whose vectors point in the same direction as its own. In a trained model the $\mathbf{W}$ matrices are learned, so "relevance" becomes whatever helps prediction.
</details>

<details>
<summary><b>Q3.</b> SASRec's training loss went from 7.96 to 6.00. Roughly how many items is it "choosing among" at each point?</summary>

$e^{7.96} \approx 2{,}900$ and $e^{6.00} \approx 400$. It narrowed its uncertainty about sevenfold.
</details>

<details>
<summary><b>Q4.</b> Why is left padding used instead of right padding?</summary>

So the most recent item is always at the last position, where the prediction is read off. With right padding, the "last real item" would sit at a different position for each user.
</details>

---

**Next:** [Part 3, Text embeddings and semantic kNN](03_text_embeddings.md): giving the model knowledge about what items *are*, not just who bought them.
