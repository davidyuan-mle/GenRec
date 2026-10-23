# Part 6: Generative Retrieval with TIGER

> **After this part you should be able to explain:** how a user's history becomes a sequence of code tokens, how an encoder-decoder Transformer reads that history and writes the next item's codes, how it's trained (teacher forcing), how beam search plus a trie turn its outputs into real items, and the engineering it took to make it run on a laptop.

This is Part 0's **Approach 2**, built from the pieces of Parts 2–5. **TIGER** (Rajput et al., 2023) is the paper that introduced this recipe.

---

## 6.1 The whole idea in one picture

```
history items   Jasmin oil            Black Opium oil
Semantic IDs    (233,242,229,0)       (233,49,184,0)
tokens          234 499 742 769       234 306 697 769        ← 4 tokens per item
                └──────────────── encoder ───────────────┘
                                    │  (memory: one vector per token)
                                    ▼
decoder   input:  [BOS]   233   242   229          ← codes generated so far
          output:  233 →  242 → 229 →  2           ← next item's codes, one at a time
                                                     = (233,242,229,2) = Roses oil
```

The **encoder** reads the whole history. The **decoder** writes the next item's Semantic ID one code at a time, exactly the chain rule from Part 0:

$$
P(\text{next item} \mid h) = \prod_{l=1}^{4} P(c_l \mid c_1, \dots, c_{l-1}, h)
$$

---

## 6.2 Turning codes into tokens

### The math

A Transformer works on **tokens**: integer IDs that each get an embedding vector. The 4 code levels must not share token IDs: "233 at level 1" (a perfume oil) has nothing to do with "233 at level 2". So each level gets its own range:

$$
\text{token}(l, c) = 1 + l \cdot K + c, \qquad l \in \lbrace 0, 1, 2, 3\rbrace ,\quad  c \in \lbrace 0, \dots, K-1\rbrace ,\quad  K = 256
$$

| Token range | Meaning |
|---|---|
| 0 | padding |
| 1 – 256 | level-1 codes |
| 257 – 512 | level-2 codes |
| 513 – 768 | level-3 codes |
| 769 – 1,024 | level-4 (tie-breaker) codes |
| 1,025 | BOS ("begin sequence", the decoder's start signal) |

That's **1,026 tokens in total**, compared with SASRec's 12,102-row item table. That small, fixed vocabulary is Part 0's main selling point.

### Worked example

Jasmin perfume oil has Semantic ID $`(233, 242, 229, 0)`$:

| Level $`l`$ | Code $`c`$ | $`1 + l \cdot 256 + c`$ | Token |
|---|---|---|---|
| 0 | 233 | $`1 + 0 + 233`$ | 234 |
| 1 | 242 | $`1 + 256 + 242`$ | 499 |
| 2 | 229 | $`1 + 512 + 229`$ | 742 |
| 3 | 0 | $`1 + 768 + 0`$ | 769 |

A history of $`n`$ items becomes $`4n`$ tokens. We keep the last 20 items, which gives at most 80 tokens, left-padded (Part 2).

---

## 6.3 The model: encoder-decoder Transformer

### The encoder: read the history

The encoder is a stack of Transformer layers, like SASRec's (Part 2), with one difference: **no causal mask**. Every history token may look at every other history token, because the whole history is input, and nothing in it is the answer. Only padding is masked.

The input at each token position is the token's embedding plus a position embedding. Positions are counted **from the right**, so the most recent item's tokens always get the same positions, whatever the history length.

Output: one vector per history token, called the **memory** $`\mathbf{H}`$.

### The decoder: write the next item

The decoder is also a stack of Transformer layers, and each layer has **two** attention steps:

1. **Causal self-attention** over the codes generated so far, exactly like SASRec: code $`l`$ can only see codes $`< l`$.
2. **Cross-attention** to the encoder memory. This is how the decoder consults the history:

$$
\text{CrossAttention} = \text{softmax}\left(\frac{(\mathbf{Y}\mathbf{W}_Q)(\mathbf{H}\mathbf{W}_K)^\top}{\sqrt{d}}\right)\mathbf{H}\mathbf{W}_V
$$

The same formula as Part 2's attention, except that **queries come from the decoder** ($`\mathbf{Y}`$, "what I'm deciding now") while **keys and values come from the encoder** ($`\mathbf{H}`$, "what's in the history"). For example, when picking code 2, the decoder can attend strongly to the tokens of past perfume oils.

### The output heads

After the decoder, a separate linear layer per level turns the state at step $`l`$ into 256 scores, and a softmax over those gives $`P(c_l \mid c_{\lt l}, h)`$. The output is always a choice among 256 codes, never among 12,101 items.

Our settings: 4 encoder layers, 4 decoder layers, width $`d = 128`$, 4 attention heads, dropout 0.1, about **2.1 million parameters** in total.

---

## 6.4 Training: teacher forcing

### The math

For a training pair (history $`h`$, true next item with codes $`(c_1, c_2, c_3, c_4)`$), feed the decoder the **true** previous codes, not its own guesses:

```
decoder input : [BOS]  c1   c2   c3
must predict  :  c1    c2   c3   c4
```

This is **teacher forcing**. Thanks to the causal mask, one pass trains all 4 steps. The loss is the average cross-entropy over the levels:

$$
\mathcal{L} = \frac{1}{4} \sum_{l=1}^{4} -\log P(c_l \mid c_1, \dots, c_{l-1}, h)
$$

### Worked example

Suppose the model gives the true codes these probabilities:

| Level | $`P(\text{true code})`$ | $`-\log P`$ |
|---|---|---|
| 1 | 0.5 | 0.693 |
| 2 | 0.6 | 0.511 |
| 3 | 0.9 | 0.105 |
| 4 | 1.0 | 0.000 |
| | **average** | **0.327** |

Multiply by 4 and you get $`1.309 = -\log(0.5 \times 0.6 \times 0.9 \times 1.0) = -\log 0.27`$. So **the summed loss is exactly the cross-entropy of the whole item**: the chain rule again. Level 4 is often near 0: once the first 3 codes are fixed, there are usually only one or a few items left.

### Training examples: one per position

TIGER makes a separate training example for **every** position of every training sequence:

```
sequence (A, B, C, D)  →  (A) → B,   (A, B) → C,   (A, B, C) → D
```

That's **131,413 pairs** from our training sequences. Why not SASRec's "predict at every position in one pass" trick (Part 2)? Because the encoder sees the **whole** history at once, with no causal mask. To predict C from (A, B), the encoder must not see C or D, so each target needs its own encoding pass. This is one reason TIGER is slower to train than SASRec.

---

## 6.5 Inference: constrained beam search

### Beam search with log-probabilities

Exactly Part 0's procedure, with beam width $`B = 20`$:

1. Encode the history once.
2. **Step 1:** score all 256 level-1 codes, and keep the 20 with the highest log-probability.
3. **Steps 2–4:** for each of the 20 kept prefixes, score the 256 next codes, giving $`20 \times 256`$ candidates. Add each candidate's log-probability to its prefix's total, and keep the best 20.
4. After step 4, the 20 surviving 4-code tuples, best first, are the top-20 recommendations.

### The trie, as integer lookups

Before each step, codes that would leave the set of real IDs get probability 0 (log-probability $`-\infty`$). To check this quickly for thousands of prefixes at once, each prefix is turned into a single integer, as if it were a number written in base 256:

$$
\text{key}(c_1, \dots, c_l) = c_1 \cdot 256^{~l-1} + c_2 \cdot 256^{~l-2} + \dots + c_l
$$

Example: prefix $`(233, 242)`$ has key $`233 \times 256 + 242 = 59{,}890`$. For each level, the code stores a sorted list of valid prefix keys, and next to each one, a 256-entry yes/no mask of allowed next codes. Looking up a whole batch of prefixes is one sorted search (`torch.searchsorted`). At the end, the full 4-code key is looked up the same way to get the item number.

Two details:

- Masked codes are removed **after** the softmax, so the remaining probabilities aren't rescaled to sum to 1. That's Part 0's "leak", and it doesn't change the ranking.
- If fewer than 20 valid items are reachable, the leftover slots are $`-\infty`$ and return item 0 ("no item"). The unit test checks that every recommendation is a real item when enough are reachable.

---

## 6.6 Making it run on a laptop

The first benchmark of the straightforward version was too slow: **about 210 s per epoch** of training plus **about 116 s** for each validation beam search. With 50 epochs that's roughly 3 hours. Three changes helped:

| Change | Why it helps | Effect |
|---|---|---|
| **Trim padding per batch** | Most histories are short (median 6 items), but every batch was padded to 80 tokens. Columns that are padding for every row of a batch are dropped. | validation: 116 s → about 50 s |
| **Sort users by history length at inference** | Similar lengths in one batch means more columns can be trimmed. | (included above) |
| **Bucket training batches by length** | In a random batch, one long history forces long padding for everyone. Batches are now built from similar-length examples, in random order. | training: about 210 s → roughly 75–80 s per epoch (estimated: 125 s per epoch minus the validation time) |

With all three, the smoke test ran at about **125 s per epoch including validation**, so a full 50-epoch run would take at most about 1.5 hours.

Trimming only works because positions are counted from the right (6.3). Otherwise removing leading padding would shift every token's position.

---

## 6.7 Results so far: a smoke test, not a real run

We only trained for **2 epochs**, to check that the whole pipeline works:

| Epoch | Train loss (per level) | Valid Recall@10 | Valid NDCG@10 |
|---|---|---|---|
| 1 | 3.68 | 0.0088 | 0.0045 |
| 2 | 2.97 | 0.0323 | 0.0168 |

| Test split | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0114 | 0.0053 |
| **TIGER, 2 epochs** | **0.0252** | **0.0128** |
| SASRec, 50 epochs | 0.0753 | 0.0423 |

- **It works end to end:** loss falls, and validation Recall@10 nearly quadrupled between epoch 1 and 2. After 2 epochs it already beats popularity by about 2×.
- **It says nothing about how good TIGER is.** It's far from converged. SASRec was still improving at epoch 30.
- The TIGER loss is per code level, over 256 options, and isn't directly comparable with SASRec's loss over 12,101 items.

**What a full run should show, and why it's hard.** Part 4 showed that Semantic IDs alone keep only about half of the content signal (kNN 0.0593 → 0.0312). To beat SASRec, TIGER's sequence model has to recover the fine ranking from purchase patterns, just as SASRec does, while also getting the content structure "for free" through shared code prefixes.

### Simplifications compared with the TIGER paper

- **No user token.** TIGER adds a token derived from the user ID; we use only the history.
- **Smaller and shorter training.** The paper trains a T5-style model for far longer.
- **Plain Adam with a constant learning rate**, the same loop as SASRec, rather than a tuned schedule.

### In our code

- [tiger.py](../src/genrec/models/tiger.py):
  - `SemanticIDTrie`: integer prefix keys, `allowed` masks, and the `items` lookup.
  - `TIGER`: `encode` (with trimming), `decode`, `loss` (teacher forcing), `generate` (constrained beam search), and `recommend`.
- [sequences.py](../src/genrec/data/sequences.py): `NextItemPairs` (one example per position) and `BucketBatchSampler`.
- [train.py](../scripts/train.py): `run_tiger`, which uses the same `fit_with_early_stopping` loop as SASRec.
- [tiger_beauty.yaml](../configs/tiger_beauty.yaml): the settings. Run the full training with `python scripts/train.py --config configs/tiger_beauty.yaml`.
- [test_tiger.py](../tests/test_tiger.py): tests the trie, valid-only generation, the pairs dataset and the sampler.

---

## 6.8 Check yourself

<details>
<summary><b>Q1.</b> What tokens represent the Semantic ID (5, 0, 255, 1)?</summary>

Level 0: $`1 + 5 = 6`$. Level 1: $`1 + 256 + 0 = 257`$. Level 2: $`1 + 512 + 255 = 768`$. Level 3: $`1 + 768 + 1 = 770`$. Tokens: **(6, 257, 768, 770)**.
</details>

<details>
<summary><b>Q2.</b> Why does the encoder have no causal mask, while SASRec does?</summary>

TIGER's encoder only reads the history, which is all allowed input, and the answer is produced separately by the decoder. SASRec reads and predicts in the same sequence, so without a causal mask position $`t`$ could see the item it's supposed to predict.
</details>

<details>
<summary><b>Q3.</b> What is the trie key of prefix (1, 2, 3)?</summary>

$`1 \cdot 256^2 + 2 \cdot 256 + 3 = 65{,}536 + 512 + 3 = 66{,}051`$.
</details>

<details>
<summary><b>Q4.</b> Teacher forcing feeds the decoder the true codes during training, but at inference it must use its own guesses. What could go wrong?</summary>

At inference the decoder may see a prefix it never saw during training, such as a wrong first code. It was only trained on correct prefixes, so its later steps may be poor. This mismatch is called **exposure bias**. Beam search softens it by keeping 20 alternatives instead of committing to one guess.
</details>

<details>
<summary><b>Q5.</b> SASRec gets <i>n</i> − 1 training signals from one forward pass over a sequence. Why does TIGER need <i>n</i> − 1 separate examples?</summary>

TIGER's encoder attends over the entire input history in both directions. To predict item $`t`$, the input must contain only items before $`t`$, so every target needs its own differently truncated input. SASRec's causal mask makes each position see only its own prefix, so one pass covers all targets.
</details>

---

**Next:** [Part 7, Putting it together](07_putting_it_together.md): what our numbers say overall, and how industrial systems extend this recipe.
