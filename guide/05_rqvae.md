# Part 5: Semantic IDs with RQ-VAE

> **After this part you should be able to explain:** what an autoencoder is, how RQ-VAE puts residual quantization inside one, why the "straight-through" trick is needed to train through a rounding step, what the codebook and commitment losses do, what codebook collapse is and how to prevent it, and how RQ-VAE compared with k-means on our data.

This is the tokenizer used by TIGER and, with improvements, by PLUM.

---

## 5.1 Intuition: learn the space before you round it

Residual k-means (Part 4) quantizes the embeddings **as they are**. But Part 3 showed those embeddings are awkward: everything sits in a narrow cone, and the differences that matter are small.

RQ-VAE's idea: **first learn a new, smaller space that's easier to quantize, then quantize there.** It learns that space together with the codebooks, with one goal: after rounding, we should still be able to recover the original embedding.

Think of packing a suitcase. k-means packs your clothes as they are. RQ-VAE first learns to fold them so they pack well, and it learns the folding and the packing together.

---

## 5.2 Building block: the autoencoder

### The math

An **autoencoder** is two neural networks trained together:

- an **encoder** $f$ that squeezes the input into a small **latent** vector: $\mathbf{z} = f(\mathbf{x})$,
- a **decoder** $g$ that tries to rebuild the input from it: $\hat{\mathbf{x}} = g(\mathbf{z})$.

Both are trained to make the rebuild match, using the **reconstruction loss**:

$$
\mathcal{L}_{\text{recon}} = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2
$$

Because $\mathbf{z}$ is smaller than $\mathbf{x}$, the encoder must keep only what matters. Ours compresses 768 numbers to 32:

```
encoder:  768 → 512 → 256 → 128 → 32     (z, the latent)
decoder:   32 → 128 → 256 → 512 → 768    (x̂, the reconstruction)
```

Each arrow is a linear layer followed by a ReLU activation, except the last one. These sizes follow TIGER.

---

## 5.3 RQ-VAE = autoencoder + residual quantization in the middle

### The math

Insert Part 4's residual quantization between the encoder and decoder. It works on the 32-number latent, with $L = 3$ codebooks of $K = 256$ vectors each:

$$
\mathbf{z} = f(\mathbf{x}), \qquad \mathbf{r}_0 = \mathbf{z}
$$

$$
c_l = \arg\min_k \big\Vert \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_k\big\Vert , \qquad \mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_{c_l} \qquad (l = 1, 2, 3)
$$

$$
\hat{\mathbf{z}} = \mathbf{e}^{(1)}_{c_1} + \mathbf{e}^{(2)}_{c_2} + \mathbf{e}^{(3)}_{c_3}, \qquad \hat{\mathbf{x}} = g(\hat{\mathbf{z}})
$$

The item's Semantic ID is $(c_1, c_2, c_3)$, the same format as before. The difference is that the encoder, the decoder and the codebook vectors $\mathbf{e}^{(l)}_k$ are **all learned together**.

---

## 5.4 The problem: you can't take the gradient of rounding

### Why it's a problem

Neural networks learn by **gradient descent**. For each weight, compute how the loss would change if the weight changed a little (the **gradient**), then nudge the weight the way that lowers the loss.

But $\arg\min$ ("pick the nearest code") is a **step function**. Move $\mathbf{z}$ a tiny bit, and the chosen code usually doesn't change, so the output doesn't change and the gradient is **zero**. Then the encoder gets no learning signal at all.

### The fix: the straight-through estimator

Use the quantized value in the **forward pass**, but pretend quantization was the identity function in the **backward pass**:

$$
\hat{\mathbf{z}}_{\text{ST}} = \mathbf{z} + \text{sg}\big[\hat{\mathbf{z}} - \mathbf{z}\big]
$$

Here $\text{sg}[\cdot]$ means **stop-gradient**: "use this value, but treat it as a constant when computing gradients".

- **Forward:** $\mathbf{z} + (\hat{\mathbf{z}} - \mathbf{z}) = \hat{\mathbf{z}}$. The decoder sees the quantized vector.
- **Backward:** the bracket counts as a constant, so $\partial \hat{\mathbf{z}}_{\text{ST}} / \partial \mathbf{z} = 1$. The decoder's gradient passes **straight through** to the encoder.

### Worked example (1 number)

Codebook $\lbrace 0, 1, 2, 3\rbrace $. The encoder outputs $z = 2.3$, which rounds to $\hat{z} = 2$. Say the decoder is the identity, and the target is $x = 2.5$.

1. **Forward:** $\hat{x} = \hat{z} = 2$, loss $= (2 - 2.5)^2 = 0.25$.
2. **Gradient at the decoder input:** $\partial \text{loss} / \partial \hat{z} = 2(2 - 2.5) = -1$.
3. **Without straight-through:** the gradient reaching $z$ is $0$, and the encoder learns nothing.
4. **With straight-through:** the gradient reaching $z$ is $-1$. Gradient descent increases $z$, toward 2.5, which is the right direction. Once $z$ passes 2.5, it rounds to 3, and the code changes.

It's an approximation: we pretend rounding wasn't there. In practice it works well, and it's the standard trick (it comes from VQ-VAE, which RQ-VAE builds on).

---

## 5.5 The full loss: three terms

The reconstruction loss trains the encoder and decoder. Two more terms keep the latents and the codebooks close to each other. For each level $l$, with residual $\mathbf{r}_{l-1}$ and chosen code vector $\mathbf{e}_l$:

$$
\mathcal{L} = \underbrace{\Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2}_{\text{reconstruction}} + \sum_{l=1}^{L} \Big( \underbrace{\big\Vert \text{sg}[\mathbf{r}_{l-1}] - \mathbf{e}_l\big\Vert ^2}_{\text{codebook loss}} + \beta \underbrace{\big\Vert \mathbf{r}_{l-1} - \text{sg}[\mathbf{e}_l]\big\Vert ^2}_{\text{commitment loss}} \Big)
$$

| Term | What gets updated | Effect |
|---|---|---|
| Reconstruction | encoder, decoder | the codes must keep enough information to rebuild $\mathbf{x}$ |
| Codebook loss | code vectors only (the residual is frozen by sg) | moves each chosen code **toward** the residuals that picked it, like k-means' "move centroid to the mean" step |
| Commitment loss, weighted by $\beta$ | encoder only (the code is frozen by sg) | pulls the encoder's outputs **toward** their codes, so latents don't drift far from the codebook |

We use $\beta = 0.25$, as in the original VQ-VAE and TIGER. The encoder should commit to codes, but more gently than the codes move toward it.

### Worked example (1 number)

Residual $r = 2.3$, chosen code $e = 2$:

| Term | Value | Gradient | Effect of a gradient step |
|---|---|---|---|
| Codebook: $(\text{sg}[r] - e)^2$ | $0.09$ | wrt $e$: $2(e - r) = -0.6$ | $e$ moves **up**, toward 2.3 |
| Commitment: $0.25 ~ (r - \text{sg}[e])^2$ | $0.0225$ | wrt $r$: $0.25 \cdot 2(r - e) = +0.15$ | $r$ moves **down**, toward 2 |

The code and the latent move toward each other, the code faster.

---

## 5.6 Codebook collapse, and two fixes

### The problem

Only codes that **get chosen** receive a codebook-loss gradient. A code that starts far from every latent is never chosen, so it never moves, so it's never chosen. It's **dead**. Meanwhile popular codes attract ever more items. This rich-get-richer effect is **codebook collapse**: you pay for 256 codes and use a fraction of them.

### Fix 1: start the codebooks with k-means

Before training, run the untrained encoder on all items, then run Part 4's residual k-means on those latents to set each level's codebook. Every code starts right next to real data.

### Fix 2: reset dead codes

Every 50 epochs, find codes that no item uses and move each one onto a **randomly chosen current residual** at its level. A code placed exactly on a residual is immediately the nearest code for at least that item, so it gets used and starts learning. We stop resetting at epoch 2,000, so the codebooks can settle before training ends at 3,000.

### Did it matter? Our ablation (500-epoch runs)

| Run | Level-1 codes used | Level-1 perplexity | Collision rate |
|---|---|---|---|
| with resets | 256 / 256 | 234 | 3.2% |
| **without resets** | **123 / 256** | **100** | 5.2% |

Without resets, more than half of level 1 died. (Levels 2–3 stayed fully used either way.) In the very first smoke test, before any reset had run, level 1 had already dropped to 39–58 codes within 20 epochs.

---

## 5.7 One more detail: standardize the inputs

Before training, each of the 768 embedding dimensions is shifted and scaled to mean 0 and standard deviation 1:

$$
x_j \leftarrow \frac{x_j - \text{mean}_j}{\text{std}_j}
$$

Because of the anisotropy from Part 3, the raw embeddings have a large shared component and tiny differences between items. Standardizing removes the shared offset and puts every dimension on the same scale, so the reconstruction loss focuses on what distinguishes items. Reconstructions are mapped back to the original scale afterwards.

---

## 5.8 Our results

### Training

3,000 epochs (batch size 1,024, Adam, learning rate $10^{-3}$), about 8 minutes on a laptop CPU:

| Epoch | Reconstruction loss | Unique 3-code IDs | Codes used per level |
|---|---|---|---|
| 100 | 0.509 | 11,979 | 256 / 256 / 256 |
| 500 | 0.425 | 11,885 | 256 / 256 / 256 |
| 1,000 | 0.399 | 11,847 | 256 / 256 / 256 |
| 3,000 | 0.371 | 11,873 | 256 / 256 / 256 |

- The ID structure is largely settled by around epoch 100. Later epochs mostly polish the reconstruction, and a 500-epoch run matched the full run's collision rate (3.2% vs 3.5%).
- The codebook and commitment losses *rise* during training (0.11 → 0.47). The most likely cause is that the encoder's latents grow in scale over training, which inflates all distances. Collisions and code usage stayed flat, so the rise doesn't mean the codes got worse.

### RQ-VAE vs. residual k-means (both 3 × 256)

| | Residual k-means | **RQ-VAE** |
|---|---|---|
| Items sharing their ID | 11.4% | **3.5%** |
| Perplexity, levels 1 / 2 / 3 | 228 / 127 / 49 | **235 / 248 / 245** |
| Category purity, 1-code / 2-code prefix | 57% / 84% | 57% / **93%** |
| Reconstruction cosine | 0.969 | 0.971 |
| kNN on reconstructions, test Recall@10 | 0.0302 | 0.0312 |

**RQ-VAE produces better IDs.** It has a third as many collisions, and all three levels use almost all 256 codes. With k-means, level 3 was mostly unused. Two-code prefixes are noticeably purer.

**It doesn't keep more similarity detail.** kNN on its reconstructions ties with k-means, and both are about half of the raw embeddings (0.0593). The difference is within the run-to-run noise seen in Part 4. A learned tokenizer changes **how the catalog is divided into codes**, not how much fine-grained similarity survives three small integers.

Example level-1 groups: code 128 is 34 items, half of them eyebrow products ("Anastasia Brow Wiz", "Benefit Brow Pencil"). Code 45 is 49 items, mostly self-tanners and bronzers.

The remaining collisions are, again, near-duplicates: six Al-Rehab perfume oils that differ only by scent name, Essie polish shades, Red Cherry lash numbers. Part 0's real Semantic ID example came from here.

### In our code

- [rqvae.py](../src/genrec/tokenizers/rqvae.py):
  - `ResidualQuantizer`: straight-through output, codebook and commitment losses, `init_kmeans`, `reset_dead_codes`.
  - `RQVAE`: the encoder, quantizer and decoder.
  - `fit_rqvae`: the training loop, with standardization.
- [train_rqvae.py](../scripts/train_rqvae.py) with [rqvae_beauty.yaml](../configs/rqvae_beauty.yaml): writes `semantic_ids/rqvae-3x256.npy`, with the 4th tie-breaker code added.
- [test_tokenizers.py](../tests/test_tokenizers.py): checks that gradients reach the encoder through the straight-through path, and that dead codes get revived.
- [Notebook 04](../notebooks/04_semantic_ids_rqvae.ipynb): training curves, the comparison, and the ablation.

---

## 5.9 Check yourself

<details>
<summary><b>Q1.</b> In the straight-through example, what would happen to $z = 2.3$ if the target were $x = 1.8$ instead?</summary>

Forward: $\hat{x} = 2$, so the gradient at the decoder input is $2(2 - 1.8) = +0.4$. Straight-through passes $+0.4$ to $z$, so gradient descent **lowers** $z$, toward 1.8.

There's a catch. While $z$ rounds to 2, the gradient keeps saying "go lower", so $z$ can drift past 1.5 and round to 1. That's worse ($\hat{x} = 1$, error 0.8 instead of 0.2), and now the gradient pushes $z$ back up, so $z$ can bounce around the boundary. Straight-through gradients are only an approximation. The commitment loss (5.5) helps here: it pulls $z$ toward its current code, which keeps it from wandering across boundaries.
</details>

<details>
<summary><b>Q2.</b> Why does the codebook loss put stop-gradient on the residual, and the commitment loss put it on the code?</summary>

So each term moves only one side. The codebook loss should move the **codes** toward the data (like a k-means update) without dragging the encoder. The commitment loss should move the **encoder** toward the codes without moving them. The two pulls get different strengths, with $\beta = 0.25$ on the commitment side.
</details>

<details>
<summary><b>Q3.</b> A code has been dead for 100 epochs. Why won't it come back by itself?</summary>

It's never the nearest code for any residual, so it's never chosen, so the codebook loss (which only touches chosen codes) never moves it. Nothing pulls it toward the data. A reset places it directly on a real residual, which makes it the nearest code for at least that item.
</details>

<details>
<summary><b>Q4.</b> RQ-VAE has far fewer collisions than k-means but the same kNN score. What does that suggest for choosing a tokenizer?</summary>

Collisions and code usage measure how the catalog is **divided**: how distinct and balanced the IDs are, which matters for a generative model that must learn to produce them. kNN on reconstructions measures how much **similarity detail** survives, and both tokenizers lose about the same. So choose on how well a generative model learns from the IDs, which is Part 6's question. RQ-VAE's balanced, low-collision codes are the usual choice.
</details>

---

**Next:** [Part 6, Generative retrieval with TIGER](06_generative_retrieval.md): training a model to write out the next item's Semantic ID.
