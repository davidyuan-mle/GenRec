# GenRec — Learning Generative Recommendation

A personal, ongoing learning repo on **generative recommendation (GenRec)**: recommender systems that *generate* the items to recommend, usually as sequences of discrete tokens decoded by a Transformer or LLM, instead of scoring candidates pulled from an embedding index.

The goal is to build a working understanding of the field by reading industrial papers and tech blogs, writing notes, and over time adding small experiments.

## What this repo covers

Topics I'm tracking:

- **Item tokenization / Semantic IDs** — turning items into short sequences of discrete codes (e.g., RQ-VAE) so a language model can generate them
- **Generative retrieval** — autoregressively decoding the Semantic IDs of the next item(s) from user context
- **Adapting pre-trained LLMs** — continued pre-training, fine-tuning, and aligning item tokens with text
- **Reasoning in recommendation** — explicit, controllable reasoning before recommending
- **Production concerns** — latency, cost, serving at scale, and how online A/B results compare with offline metrics

## Reading list

| Paper / Post | Org | Year | File |
|---|---|---|---|
| PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations | Google DeepMind / YouTube | 2025 | [PLUM-2025.pdf](PLUM-2025.pdf) |
| OneRec-Think: In-Text Reasoning for Generative Recommendation | Kuaishou | 2025 | [OneRecThink-2025.pdf](OneRecThink-2025.pdf) |
| GLIDE: Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify | Spotify | 2026 | [GLIDE-2026.pdf](GLIDE-2026.pdf) |
| GenRec: Towards LLM-Native Recommendation at Netflix | Netflix | 2026 | [Medium post](https://medium.com/netflix-techblog/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) |

### PLUM (YouTube, 2025)

A framework for adapting a pre-trained LLM to industry-scale recommendation, in three stages:

1. **Item tokenization** with Semantic IDs (plus several improvements to them)
2. **Continued pre-training (CPT)** on domain-specific data
3. **Task-specific fine-tuning**, focused on **generative retrieval**: the model is trained to generate the Semantic IDs of recommended items from user context

On YouTube's large-scale video data, PLUM substantially improves retrieval over a heavily optimized production model built on large embedding tables. The paper also includes a scaling study, lessons learned about CPT, and the training and inference setup used to launch it to billions of users.

**Why read it:** a clear, end-to-end blueprint for turning a general LLM into a production generative retriever.

### OneRec-Think (Kuaishou, 2025)

Generative recommenders like OneRec act as *implicit* predictors and can't reason explicitly. OneRec-Think combines dialogue, reasoning, and recommendation in a single model through three parts:

1. **Itemic alignment** — aligns item tokens with text so the model can describe items and user preferences in natural language
2. **Reasoning activation** — "reasoning scaffolding" that gets the LLM to reason inside the recommendation context
3. **Reasoning enhancement** — RL with a recommendation-specific reward that accounts for the fact that many different items can be valid recommendations for a user

It reaches state-of-the-art results on public benchmarks. A **"Think-Ahead"** architecture made industrial deployment practical, and it gave a **+0.159% gain in app stay time** on Kuaishou.

**Why read it:** shows how chain-of-thought-style reasoning can be brought into a Semantic-ID generative recommender, and what it takes to serve it.

### GLIDE (Spotify, 2026)

A production generative recommender for **podcast discovery**. It treats recommendation as an **instruction-following task** over a catalog discretized into Semantic IDs, so every generated item exists in the real inventory. The model takes recent listening history and lightweight user context as input, and adds **long-term user embeddings as soft prompts** to capture stable preferences while staying within strict inference limits.

It was evaluated with offline retrieval metrics, human judgments, and LLM-based evaluation, then in online A/B tests with millions of users: **up to +5.4% non-habitual podcast streaming** and **up to +14.3% new-show discovery** on the Spotify home surface, while meeting cost and latency targets.

**Why read it:** a practical deployment case study covering grounding, personalization through soft prompts, discovery-focused objectives, and a multi-angle evaluation approach.

### Netflix GenRec (Netflix Tech Blog, Jul 2026)

An **LLM-backed ranker** built by post-training Netflix's internal foundation LLM. It is meant to replace a production ranker that relies on thousands of hand-crafted features. Unlike the three papers above, it does **not** decode Semantic IDs. A **catalog-aware scoring head** scores every Netflix title from a pooled hidden state, so recommendations can only come from the real catalog.

- **Two-phase training:** Phase 1 adapts an open-source LLM on Netflix data into a shared foundation model and is updated infrequently. Phase 2 post-trains it for ranking and is refreshed often.
- **Logs → conversations:** user history, context and item metadata are *verbalized* as text. The user's actual engagement becomes the "assistant" reply.
- **Context engineering replaces feature engineering:** the token budget is the new feature budget. Keep high-signal events, drop low-signal ones, compress binges and expand on new or cold-start titles. Context was cut to about **one-third of the tokens** with negligible quality loss.
- **Multi-objective loss:** cross-entropy ranking over the catalog, plus an LM loss, plus **reward-weighted** examples for long-term satisfaction and content-type rebalancing. This is a cheaper alternative to full RL such as GRPO.
- **Serving:** vLLM in **prefill-only** mode scores all candidates in one forward pass with no token-by-token decoding. Smaller or distilled backbones keep cost down.
- **Results:** offline, **+1.6% MRR with about 40× fewer Phase-2 labels**. Online, an A/B test on about 10% of traffic for about 4 weeks showed significant gains on both short-term engagement (+0.115%) and long-term metrics (+0.006%). In ablations, Phase 1 adds 10–20% over an open-source base, and Phase 2 adds 35–50%, growing to about 80% as Phase 1 gets two weeks stale. Quality improves steadily with more data and larger models (about 1B to 10B parameters).

**Why read it:** a different design point from Semantic-ID generation (LLM as a *ranker* with a scoring head). It is also a clear statement of the "LLM-native RecSys" shift: prompts instead of feature vectors, shared foundation backbones, scaling laws as a design guide, and LLM serving infrastructure.

## Repo structure

```
GenRec/
├── README.md
└── *.pdf          # papers (named <Name>-<Year>.pdf)
```

Planned additions: per-paper notes, a cross-paper comparison (tokenization, training recipe, serving, metrics), and small hands-on experiments (e.g., a toy RQ-VAE Semantic ID tokenizer plus a generative retriever).
