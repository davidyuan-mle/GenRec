"""Companion to guide/08_training_walkthrough.md.

Runs the repo's real data and training code on a toy batch (6 items, 3 users,
hidden size 4) and prints every intermediate value the guide walks through.

  python guide/walkthrough_training_step.py

The first block needs the processed data (python scripts/prepare_data.py).
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genrec.data.sequences import NextItemDataset, NextItemPairs, leave_one_out
from genrec.models.sasrec import SASRec
from genrec.models.tiger import TIGER

torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
names = {0: "PAD", 1: "shampoo", 2: "conditioner", 3: "hair oil", 4: "red polish", 5: "top coat", 6: "perfume"}

print("=== real-scale counts")
seqs = json.load(open("data/processed/beauty/sequences.json"))
train, valid, test = leave_one_out(seqs)
ds = NextItemDataset(train, 50)
signals = sum(min(len(s), 51) - 1 for s in ds.seqs)
print("train sequences", len(ds), "batches/epoch at 256", -(-len(ds) // 256), "training signals/epoch", signals)
print("TIGER pairs", len(NextItemPairs(train, 20)))
print("logits tensor per batch", 256 * 50 * 12102, "floats =", 256 * 50 * 12102 * 4 / 1e6, "MB")

print("\n=== toy data")
toy = {"Ann": [1, 2, 3, 4, 5], "Ben": [4, 5, 1, 2, 6, 3], "Cat": [6, 1, 2, 3, 4]}
tr, va, te = leave_one_out(list(toy.values()))
for (u, s), t, vh, vt, th, tt in zip(toy.items(), tr, va[0], va[1], te[0], te[1]):
    print(u, "full", s, "| train", t, "| valid", vh, "->", vt, "| test", th, "->", tt)
tds = NextItemDataset(tr, max_len=4)
for i in range(len(tds)):
    x, y = tds[i]
    print("example", i, "inputs", x.tolist(), "targets", y.tolist())
inputs, targets = next(iter(DataLoader(tds, batch_size=3, shuffle=False)))
print("batch inputs\n", inputs, "\nbatch targets\n", targets)

print("\n=== toy SASRec forward")
torch.manual_seed(0)
model = SASRec(n_items=6, max_len=4, hidden=4, n_layers=1, n_heads=1, dropout=0.0)
print("params", sum(p.numel() for p in model.parameters()))
E, P = model.item_emb.weight.detach(), model.pos_emb.weight.detach()
print("item_emb E\n", E, "\npos_emb P\n", P)
x_ann = E[inputs[0]] + P
print("Ann x = E[inputs] + P\n", x_ann)
h = model.encode(inputs)
print("Ann hidden h (after attention + norm)\n", h[0].detach())
logits = model.logits(h)
logits.retain_grad()
print("logits shape", tuple(logits.shape))
print("Ann logits at last position (predicting after 'conditioner')\n", logits[0, 3].detach())
p = F.softmax(logits[0, 3].detach(), -1)
print("softmax", p, "P(target=3 hair oil)", p[3].item(), "-log", -torch.log(p[3]).item())
per = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), ignore_index=0, reduction="none").view(3, 4)
print("per-position loss (0 where target is PAD)\n", per.detach())
loss = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), ignore_index=0)
print("non-pad targets", (targets != 0).sum().item(), "mean loss", loss.item(), "uniform ln 7", np.log(7))
assert abs(loss.item() - model.loss(inputs, targets).item()) < 1e-6

print("\n=== backward")
model.zero_grad()
loss.backward()
g = logits.grad[0, 3]
onehot = F.one_hot(torch.tensor(3), 7).float()
print("grad of Ann last-position logits", g)
print("(softmax - onehot) / n_nonpad", (p - onehot) / 7)
print("grad on item_emb rows (norm per row)", model.item_emb.weight.grad.norm(dim=1))
w = model.encoder.layers[0].self_attn.in_proj_weight
print("one attention weight before", w[0, 0].item(), "grad", w.grad[0, 0].item())

print("\n=== Adam step")
opt = torch.optim.Adam(model.parameters(), lr=0.001)
before = {n: q.detach().clone() for n, q in model.named_parameters()}
opt.step()
deltas = torch.cat([(q.detach() - before[n]).flatten() for n, q in model.named_parameters()])
print("weight after", w[0, 0].item(), "change", (w[0, 0] - before["encoder.layers.0.self_attn.in_proj_weight"][0, 0]).item())
print("abs change: median", deltas.abs().median().item(), "max", deltas.abs().max().item(),
      "share of params changed by ~lr (0.0009-0.0011)", ((deltas.abs() > 0.0009) & (deltas.abs() < 0.0011)).float().mean().item())
with torch.no_grad():
    print("loss on same batch after 1 step", model.loss(inputs, targets).item())
for step in range(2, 201):
    opt.zero_grad(); l = model.loss(inputs, targets); l.backward(); opt.step()
    if step in (10, 50, 200):
        print(f"loss after {step} steps", l.item())
with torch.no_grad():
    pr = F.softmax(model.logits(model.encode(inputs))[0, 3], -1)
print("Ann last-position probs after 200 steps", pr, "P(hair oil)", pr[3].item())

print("\n=== toy TIGER batch")
sem = np.array([[-1, -1, -1], [0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
torch.manual_seed(0)
tiger = TIGER(sem, max_items=3, d_model=8, n_heads=2, n_layers=1, dim_ff=16, dropout=0.0, codebook_size=4)
print("item_tokens table\n", tiger.item_tokens, "BOS", tiger.bos)
pairs = NextItemPairs(tr[:1], max_len=3)
for i in range(len(pairs)):
    hx, tx = pairs[i]
    print("pair", i, "history", hx.tolist(), "target", tx.item())
hist, tgt = next(iter(DataLoader(pairs, batch_size=2)))
print("encoder tokens\n", tiger.item_tokens[hist].flatten(1))
codes = tiger.item_tokens[tgt] - tiger.offsets
print("target codes", codes.tolist())
print("decoder input tokens = [BOS] + first two codes as tokens:", [[tiger.bos] + (c[:-1] + tiger.offsets[:-1]).tolist() for c in codes])
memory, pad = tiger.encode(hist)
hdec = tiger.decode(memory, pad, codes[:, :-1])
per_level = [F.cross_entropy(head(hdec[:, l]), codes[:, l]).item() for l, head in enumerate(tiger.heads)]
print("memory shape", tuple(memory.shape), "decoder states", tuple(hdec.shape))
print("per-level loss", per_level, "mean", np.mean(per_level), "ln 4", np.log(4), "model.loss", tiger.loss(hist, tgt).item())
