# Cross-Paper Comparison

| | PLUM | OneRec-Think | GLIDE | Netflix GenRec |
|---|---|---|---|---|
| Org / domain | YouTube video | Kuaishou short video | Spotify podcasts | Netflix titles |
| Task | Generative retrieval | Generative rec + reasoning | Generative retrieval | Ranking |
| Item representation | Semantic IDs | Semantic IDs (itemic tokens) | Semantic IDs | Text + learned item embeddings |
| Backbone | Pre-trained LLM + CPT | LLM | LLM | Netflix-adapted foundation LLM |
| Training | CPT → SFT | Alignment → reasoning SFT → RL | Instruction tuning | Phase 1 → Phase 2 (ranking + LM + reward-weighted) |
| Personalization | User context sequence | Interaction history | Recent history + soft-prompt user embeddings | Verbalized history + context |
| Catalog grounding | Valid Semantic IDs | Valid Semantic IDs | Semantic ID catalog | Catalog-aware scoring head |
| Serving | | Think-Ahead | Latency-constrained | vLLM, prefill-only |
| Headline result | Beats embedding-table prod retriever | +0.159% app stay time | +5.4% non-habitual streams, +14.3% new-show discovery | +1.6% MRR with ~40× fewer labels; online wins |
