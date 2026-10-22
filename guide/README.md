# Learning Guide: Generative Recommendation from Scratch

A beginner-friendly walkthrough of everything built in this repo, written to be read in order. Each part follows the same pattern:

1. **Intuition**: what problem this piece solves, in plain words
2. **The math**: the few formulas that define it, with every symbol explained
3. **Worked example**: tiny numbers you can check by hand
4. **In our code**: where it lives in `src/genrec/` and what our experiments showed
5. **Check yourself**: a few questions with hidden answers

No prior recommender-systems knowledge is assumed. You need high-school algebra, and some comfort with vectors (lists of numbers) and probabilities. Everything else is introduced when it's needed.

## Contents

| Part | Topic | Milestone | Status |
|---|---|---|---|
| [0](00_big_picture.md) | **The big picture**: what generative recommendation is and why anyone would do it | all | ✅ draft |
| 1 | Data and evaluation: from reviews to sequences, leave-one-out, Recall@K / NDCG@K / MRR | 1 | ⬜ |
| 2 | Baselines: popularity and SASRec (self-attention for sequences) | 1 | ⬜ |
| 3 | Text embeddings and semantic kNN: turning product text into vectors | 2 | ⬜ |
| 4 | Semantic IDs I: k-means, residual quantization, collisions | 3 | ⬜ |
| 5 | Semantic IDs II: RQ-VAE (learned quantization, straight-through gradients, codebook collapse) | 4 | ⬜ |
| 6 | Generative retrieval: TIGER (encoder-decoder, teacher forcing, constrained beam search) | 5 | ⬜ |
| 7 | Putting it together: what our numbers say, and how PLUM / OneRec / GLIDE / Netflix extend this | — | ⬜ |

## Notation used throughout

| Symbol | Meaning |
|---|---|
| $\mathcal{I}$, $N$ | the set of all items (the catalog), and how many there are ($N = 12{,}101$ in our data) |
| $h = (i_1, \dots, i_n)$ | one user's history: the items they interacted with, oldest first |
| $\mathbf{v}$ (bold) | a vector, e.g. $\mathbf{v}_i \in \mathbb{R}^d$ is item $i$'s vector of $d$ numbers |
| $\mathbf{a} \cdot \mathbf{b}$ | dot product: multiply matching entries and add them up |
| $P(A \mid B)$ | probability of $A$ given that we know $B$ |
| $(c_1, \dots, c_L)$ | a Semantic ID: $L$ discrete codes, each a whole number from $0$ to $K-1$ |
