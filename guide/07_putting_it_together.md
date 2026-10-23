# Part 7: Putting It Together

> **After this part you should be able to explain:** what all our experiments say when read side by side, which lessons carry over to real systems, what the obvious next experiments are, and how the four industrial systems in the reading list build on (or depart from) this recipe.

---

## 7.1 All the numbers in one place

Test split, Amazon Beauty, ranking over all 12,101 items:

| Model | Part | Approach | Learns from | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| Random guess | 1 | — | nothing | ≈ 0.0008 | | |
| Popularity | 2 | score | counts | 0.0114 | 0.0053 | 0.0040 |
| Semantic kNN on k-means IDs | 4 | score | text, quantized | 0.0302 | 0.0156 | 0.0124 |
| Semantic kNN on RQ-VAE IDs | 5 | score | text, quantized | 0.0312 | 0.0164 | 0.0131 |
| Semantic kNN, raw embeddings | 3 | score | text | 0.0593 | 0.0350 | 0.0290 |
| **SASRec** | 2 | score | purchase sequences | **0.0753** | **0.0423** | **0.0346** |
| TIGER, 2-epoch smoke test | 6 | **generate** | sequences of RQ-VAE IDs | 0.0252 | 0.0128 | — |

The TIGER row only shows that the pipeline works. It isn't a fair comparison until it's trained properly.

---

## 7.2 Five lessons

**1. Measure against a floor.** Popularity is about 14× better than random, just because the catalog's popularity is skewed. Without it, "0.03" would be impossible to interpret.

**2. Content and behaviour are different signals.** Text similarity alone reaches about 80% of SASRec with no training, and it's stable from validation to test. But it can't see **complements** (shampoo → conditioner), which is why SASRec pulls further ahead at K = 20. Good systems need both.

**3. Semantic IDs are a vocabulary, not a similarity measure.** Squeezing a 768-number embedding into 3 small codes keeps the coarse structure (57% category purity at the first code, 93% at two codes with RQ-VAE) but halves the fine-grained retrieval signal. The rounding error is as large as the distance between neighbouring items. The sequence model on top has to supply the fine ranking.

**4. A good tokenizer is about balance, not just accuracy.** RQ-VAE didn't keep more similarity detail than k-means, but it produced far fewer collisions (3.5% vs 11.4%) and used all codes at all levels. For a model that has to *generate* codes, a balanced vocabulary is what matters. Without dead-code resets, half of level 1 died.

**5. Generative retrieval costs more compute.** TIGER needs one training example per position (131k pairs) instead of one pass per sequence, and a 4-step beam search per user instead of one matrix multiply. Laptop-scale TIGER needed padding trimming and length bucketing just to train in reasonable time. At industrial scale, this cost is what papers spend much of their effort on.

---

## 7.3 Natural next experiments

Each of these is a small, self-contained follow-up to the code in this repo:

| Experiment | Question it answers |
|---|---|
| Full TIGER run (`configs/tiger_beauty.yaml`) | Can generative retrieval match SASRec here? |
| TIGER on k-means IDs vs RQ-VAE IDs | Does the cleaner RQ-VAE vocabulary actually help generation? |
| Cold-start test: score only test targets that are rare in training | Does the content-derived ID help where SASRec is weakest (e.g. the 33 never-seen test items)? |
| Add a user token, as TIGER does | How much does knowing the user (not just the history) add? |
| Combine scores (SASRec + semantic kNN) | Are collaborative and content signals complementary in practice? |

---

## 7.4 How industry builds on this (the reading list)

The repo's reading list covers four systems. Here is where each one sits relative to what we built. These summaries come from the README's paper notes, so see the papers for details.

### PLUM (Google / YouTube, 2025): Semantic IDs inside a pre-trained LLM

PLUM is the closest relative of our pipeline, scaled up and built on a **pre-trained language model**:

| Our step | PLUM's version |
|---|---|
| RQ-VAE Semantic IDs (Part 5) | Semantic IDs with several improvements to the tokenizer |
| (none) | **Continued pre-training** of an LLM on domain data, so the LLM learns the "language" of Semantic IDs alongside text |
| TIGER fine-tuning (Part 6) | **Task-specific fine-tuning for generative retrieval**: generate Semantic IDs of recommended videos from user context |

The key difference: our TIGER starts from random weights and knows nothing but our sequences. PLUM starts from an LLM that already knows language and the world, and teaches it the item codes. At YouTube scale it beat a heavily optimized production model built on large embedding tables, which is exactly Part 0's Approach 1.

### OneRec-Think (Kuaishou, 2025): reasoning before recommending

A Semantic-ID generator like TIGER is an **implicit** predictor: it outputs codes with no explanation. OneRec-Think adds explicit reasoning:

- **Itemic alignment:** teach the model to connect item codes with text, so it can describe items and preferences in words.
- **Reasoning activation:** get the model to reason (chain-of-thought style) inside the recommendation context.
- **Reasoning enhancement:** reinforcement learning with a reward designed for recommendation, which accounts for the fact that many different items can be good answers.

In our terms: before the decoder writes codes (Part 6), it first writes a few sentences of reasoning about the user. Deployed with a "Think-Ahead" architecture to meet latency limits, it reported +0.159% app stay time.

### GLIDE (Spotify, 2026): instruction-following over Semantic IDs

GLIDE applies the recipe to **podcast discovery**:

- Recommendation as an **instruction-following** task over a catalog tokenized into Semantic IDs. Grounding in real IDs guarantees every output exists, which is our trie's job in Part 6.
- Inputs: recent listening history plus lightweight context, and **long-term user embeddings injected as soft prompts**: vectors fed to the model as if they were extra tokens. This is a more powerful version of the "user token" our TIGER leaves out.
- Objectives aimed at **discovery** (new shows, non-habitual listening). Online it reported up to +14.3% new-show discovery.

### Netflix GenRec (2026): an LLM ranker without Semantic IDs

Netflix takes a **different design point**. It does *not* generate Semantic IDs:

- User history, context and item metadata are written out as **text conversations**.
- A post-trained LLM reads them, and a **catalog-aware scoring head** scores every title from one pooled hidden state.

That's Part 0's **Approach 1** (score every item), but with an LLM as the history encoder instead of SASRec's small Transformer. Since outputs are scores over real titles, there's nothing to hallucinate, so no trie is needed. Their lessons are about context engineering ("the token budget is the new feature budget"), cheap reward-weighted training instead of full RL, and serving with prefill-only scoring.

### The landscape in one table

| System | Item representation | History encoder | Output | Closest part of this repo |
|---|---|---|---|---|
| SASRec | one learned vector per item | small Transformer | score every item | Part 2 |
| TIGER | Semantic IDs (RQ-VAE) | encoder-decoder from scratch | generate codes | Part 6 |
| PLUM | improved Semantic IDs | pre-trained LLM | generate codes | Parts 5–6 + LLM |
| OneRec-Think | Semantic IDs aligned with text | LLM that reasons first | reasoning, then codes | Part 6 + reasoning |
| GLIDE | Semantic IDs | LLM with user soft prompts | generate codes | Part 6 + user context |
| Netflix GenRec | text metadata | post-trained LLM | score every title | Part 2's approach, LLM-sized |

---

## 7.5 Check yourself

<details>
<summary><b>Q1.</b> Which of our results is the strongest evidence that Semantic IDs lose information?</summary>

Semantic kNN drops from Recall@10 0.0593 on raw embeddings to about 0.03 on Semantic-ID reconstructions, for both k-means and RQ-VAE. Same recommender, same data; only the quantization changed.
</details>

<details>
<summary><b>Q2.</b> Netflix's system scores every title like SASRec does. So why is it still called "generative recommendation"?</summary>

Its history encoder is a generative LLM, post-trained on verbalized logs, and the inputs are text prompts instead of feature vectors. What makes it "LLM-native" is how it reads the user. How it produces outputs is the same as Approach 1: it doesn't generate item identifiers. The field uses "generative recommendation" for both styles.
</details>

<details>
<summary><b>Q3.</b> What does PLUM's continued pre-training add that our TIGER lacks?</summary>

World and language knowledge. Our TIGER learns only from about 131k (history, next item) pairs, starting from random weights. PLUM starts from an LLM that already understands text, and continued pre-training teaches it how Semantic IDs relate to that knowledge, before any recommendation fine-tuning.
</details>

---

**Next:** [Part 8, One training step, end to end](08_training_walkthrough.md): follow one real user and one batch through the actual training code.
