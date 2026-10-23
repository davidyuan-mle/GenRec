# Part 8: One Training Step, End to End

> **After this part you should be able to trace:** one real user from a line in the raw review file all the way into a training batch, and one batch through a complete training step (forward pass, loss, gradients, weight update), naming the exact line of code that does each step.

Parts 1–6 explained each piece on its own. This part connects them. It works in two halves:

1. **Real data (8.1–8.5):** follow user #17 through every data-preparation stage, from the raw file to a batch.
2. **Toy model (8.6–8.11):** a model small enough to print every number, put through a complete training step **by the repo's actual code**.

Every number in this part was printed by [walkthrough_training_step.py](walkthrough_training_step.py). Run `python guide/walkthrough_training_step.py` to see them yourself.

---

## 8.0 The map

| Stage | Code | What user #17 looks like | Full-data size |
|---|---|---|---|
| 1. Raw file | `data/raw/reviews_Beauty_5.json.gz` | 5 lines of JSON | 198,502 lines |
| 2. Table | `load_reviews` | 5 rows of (user, ASIN, time) | 198,502 rows |
| 3. Sequences | `build_dataset` | `[8446, 9194, 9195, 10258, 10347]` | 22,363 lists |
| 4. Split | `leave_one_out` | train `[8446, 9194, 9195]` | 22,363 train / valid / test |
| 5. Examples | `NextItemDataset` | inputs `[0, …, 0, 8446, 9194]`, targets `[0, …, 0, 9194, 9195]` | 22,363 examples |
| 6. Batches | `DataLoader` | one row of a 256 × 50 tensor | 88 batches per epoch |
| 7. Training step | `model.loss` → `backward` → `opt.step` | (toy model below) | 88 steps per epoch |

---

## 8.1 Stage 1: the raw file

Each line of `reviews_Beauty_5.json.gz` is one review, stored as JSON. Here are user #17's five lines, with the reviewer's name removed and the review text shortened:

```json
{"reviewerID": "A0614215ICIJ7IVV472R", "asin": "B005NOPAAS", "helpful": [0, 0], "reviewText": "I love how this product detangles my hair and protects it. …", "overall": 5.0, "summary": "Awesome!", "unixReviewTime": 1358812800, "reviewTime": "01 22, 2013"}
{"reviewerID": "A0614215ICIJ7IVV472R", "asin": "B006Z8AP6K", ..., "summary": "Awesome!", "unixReviewTime": 1358812800}
{"reviewerID": "A0614215ICIJ7IVV472R", "asin": "B006Z8B5HS", ..., "summary": "Great shampoo!", "unixReviewTime": 1358812800}
{"reviewerID": "A0614215ICIJ7IVV472R", "asin": "B0091W3622", ..., "summary": "Love this conditioner!", "unixReviewTime": 1358812800}
{"reviewerID": "A0614215ICIJ7IVV472R", "asin": "B009B7E450", ..., "overall": 2.0, "summary": "Okay", "unixReviewTime": 1358812800}
```

**We keep only three fields** and throw away the rest: the star rating, the review text, the helpfulness votes. That's [amazon.py:35-42](../src/genrec/data/amazon.py#L35-L42):

```python
def load_reviews(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], r["unixReviewTime"]))
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])
```

Notice the last review: 2 stars, "Okay". The user *didn't like* that product, but it still counts as an interaction. Our models learn "what people interact with next", not "what they like".

> **A fact about this file:** the `_5` in its name means SNAP already applied the 5-core filter (Part 1). Its counts (198,502 reviews, 22,363 users, 12,101 items) match our processed data exactly. So our `kcore_filter` re-checks the condition but removes nothing on this file.

---

## 8.2 Stage 2: sort, renumber, group

[amazon.py:98-105](../src/genrec/data/amazon.py#L98-L105) does three things:

```python
df = kcore_filter(reviews, k)
# Timestamps have day resolution; a stable sort keeps file order for ties.
df = df.sort_values("timestamp", kind="stable")              # (a) sort by time

asins = sorted(df["item"].unique())
item_index = {a: i + 1 for i, a in enumerate(asins)}         # (b) ASIN -> 1..N
df["item_idx"] = df["item"].map(item_index)
sequences = df.groupby("user", sort=True)["item_idx"].apply(list).tolist()   # (c) one list per user
```

**(b) Renumbering.** Models need small integers, not strings like `B006Z8AP6K`. Sort all 12,101 ASINs alphabetically and number them from 1, keeping 0 free for padding:

| ASIN | Item index | Product |
|---|---|---|
| B005NOPAAS | 8446 | It's a 10 Keratin Leave-In Conditioner |
| B006Z8AP6K | 9194 | It's a 10 Miracle Oil Plus Keratin |
| B006Z8B5HS | 9195 | It's a 10 Miracle Shampoo Plus Keratin |
| B0091W3622 | 10258 | It's a 10 Miracle Deep Conditioner Plus Keratin |
| B009B7E450 | 10347 | It's a 10 Miracle Leave-In Potion Plus Keratin |

**(c) Grouping** gives user #17's sequence: **`[8446, 9194, 9195, 10258, 10347]`**. Users are sorted by reviewer ID, and this one happens to be the 18th (index 17). That's the row stored in `data/processed/beauty/sequences.json`.

### ⚠️ An important caveat: same-day ties

All five of user #17's reviews have the same timestamp, `1358812800`, which is 22 Jan 2013. Amazon timestamps only record the **day**. So the sort in (a) can't order them, and the stable sort keeps the **file order**, which turns out to be **ASIN order** for every user in this file.

How common is this across the whole dataset?

| Measure | Value |
|---|---|
| Consecutive items in a history that share the same day | **43.9%** |
| Users whose entire history is a single day | 8.4% |

So nearly half of all "A came before B" facts in our sequences are really "A has a smaller product ID than B". Products from one product line often have nearby IDs (see 9194 and 9195 above). A sequential model can partly learn this ID ordering instead of real behaviour. That may make next-item prediction look easier than it really is, for SASRec and TIGER alike, since they train and test on the same ordering. It's a known quirk of this benchmark, worth keeping in mind before trusting the absolute numbers.

---

## 8.3 Stage 3: leave-one-out

[sequences.py:14-19](../src/genrec/data/sequences.py#L14-L19):

```python
def leave_one_out(sequences):
    train = [s[:-2] for s in sequences]
    valid = ([s[:-2] for s in sequences], [s[-2] for s in sequences])
    test = ([s[:-1] for s in sequences], [s[-1] for s in sequences])
    return train, valid, test
```

For user #17:

| Split | Input | Target |
|---|---|---|
| train | `[8446, 9194, 9195]` | (each next item within this list) |
| valid | `[8446, 9194, 9195]` | `10258` (Deep Conditioner) |
| test | `[8446, 9194, 9195, 10258]` | `10347` (Leave-In Potion) |

---

## 8.4 Stage 4: from a sequence to a training example

[sequences.py:28-45](../src/genrec/data/sequences.py#L28-L45):

```python
class NextItemDataset(Dataset):
    def __init__(self, train_seqs, max_len):
        self.seqs = [s for s in train_seqs if len(s) >= 2]     # need at least one (input, next) pair
        self.max_len = max_len

    def __getitem__(self, idx):
        s = self.seqs[idx][-(self.max_len + 1):]                # keep the most recent max_len + 1 items
        inputs = left_pad(s[:-1], self.max_len)                 # all but the last
        targets = left_pad(s[1:], self.max_len)                 # all but the first: shifted by one
        return torch.tensor(inputs), torch.tensor(targets)
```

User #17's training list `[8446, 9194, 9195]` with `max_len = 50` becomes:

```
position :   0    1   …   47     48     49
inputs   :   0    0   …    0   8446   9194
targets  :   0    0   …    0   9194   9195
```

This one example carries **two** training signals: "after 8446, predict 9194" (position 48) and "after 8446, 9194, predict 9195" (position 49). Positions whose target is 0 (padding) are ignored by the loss.

Across the dataset: 22,363 examples (every user has at least 3 training items), carrying **128,031** training signals per epoch.

---

## 8.5 Stage 5: batches

`DataLoader(dataset, batch_size=256, shuffle=True)` in [train.py](../scripts/train.py) draws 256 random examples and **stacks** them:

```
inputs  : 256 × 50 tensor of item indices
targets : 256 × 50 tensor of item indices
```

That's ⌈22,363 / 256⌉ = **88 batches per epoch**, in a new random order every epoch. Each batch is one training step.

Now let's watch one step happen, on a model small enough to print.

---

## 8.6 The toy setup

**Catalog:** 6 items, plus index 0 for padding.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Item | PAD | shampoo | conditioner | hair oil | red polish | top coat | perfume |

**Users** and their leave-one-out split, computed by the real `leave_one_out`:

| User | Full sequence | Train | Valid | Test |
|---|---|---|---|---|
| Ann | 1, 2, 3, 4, 5 | 1, 2, 3 | (1, 2, 3) → 4 | (1, 2, 3, 4) → 5 |
| Ben | 4, 5, 1, 2, 6, 3 | 4, 5, 1, 2 | (4, 5, 1, 2) → 6 | (4, 5, 1, 2, 6) → 3 |
| Cat | 6, 1, 2, 3, 4 | 6, 1, 2 | (6, 1, 2) → 3 | (6, 1, 2, 3) → 4 |

### How the data is read in, step by step

The walkthrough script feeds these three users through the same code as real training. With `max_len = 4` (it's 50 in real training):

```python
toy = {"Ann": [1, 2, 3, 4, 5], "Ben": [4, 5, 1, 2, 6, 3], "Cat": [6, 1, 2, 3, 4]}
tr, va, te = leave_one_out(list(toy.values()))                      # step 1
tds = NextItemDataset(tr, max_len=4)                                # step 2
loader = DataLoader(tds, batch_size=3, shuffle=False)               # step 4 (shuffle=True in real training)
inputs, targets = next(iter(loader))                                # step 3 runs inside this call
```

**Step 1: `leave_one_out` keeps all but the last two items** for training: `train = [s[:-2] for s in sequences]`. That gives Ann (1, 2, 3), Ben (4, 5, 1, 2) and Cat (6, 1, 2), as in the table above.

**Step 2: `NextItemDataset.__init__` stores those lists.** It keeps only lists with at least 2 items (one to look at, one to predict). All three qualify, so `len(tds) == 3`. Nothing is converted yet: the work happens each time an example is requested.

**Step 3: `NextItemDataset.__getitem__(idx)` builds one example** ([sequences.py:41-45](../src/genrec/data/sequences.py#L41-L45)):

```python
def __getitem__(self, idx):
    s = self.seqs[idx][-(self.max_len + 1):]   # (a) keep the most recent max_len + 1 items
    inputs = left_pad(s[:-1], self.max_len)    # (b) drop the LAST item, pad on the left with 0s
    targets = left_pad(s[1:], self.max_len)    # (c) drop the FIRST item, pad on the left with 0s
    return torch.tensor(inputs), torch.tensor(targets)
```

Tracing Ben (`idx = 1`):

| Line | Code | Ben's value |
|---|---|---|
| start | `self.seqs[1]` | `[4, 5, 1, 2]` |
| (a) | last `max_len + 1 = 5` items | `[4, 5, 1, 2]` (only 4 exist) |
| (b) | `s[:-1]` | `[4, 5, 1]` |
| | `left_pad(…, 4)` | **`[0, 4, 5, 1]`** → inputs |
| (c) | `s[1:]` | `[5, 1, 2]` |
| | `left_pad(…, 4)` | **`[0, 5, 1, 2]`** → targets |

Lines (b) and (c) cut opposite ends off the same list, so the targets are the inputs **shifted one step**. Each input ends up next to the item that followed it:

```
s       :     4   5   1   2
inputs  : 0   4   5   1          ← s without its last item
targets : 0   5   1   2          ← s without its first item
              ↑   ↑   ↑
     "after 4 came 5", "after 5 came 1", "after 1 came 2"
```

The other two users, by the same steps:

| User | `s` | `s[:-1]` → inputs | `s[1:]` → targets |
|---|---|---|---|
| Ann | `[1, 2, 3]` | `[1, 2]` → `[0, 0, 1, 2]` | `[2, 3]` → `[0, 0, 2, 3]` |
| Cat | `[6, 1, 2]` | `[6, 1]` → `[0, 0, 6, 1]` | `[1, 2]` → `[0, 0, 1, 2]` |

Why `max_len + 1` in line (a)? Take a 10-item history with `max_len = 4`. Line (a) keeps the last **5** items, say `[a, b, c, d, e]`. Then the inputs are `[a, b, c, d]` and the targets are `[b, c, d, e]`: all 4 positions are used, with no padding. Keeping only 4 items would leave one position empty.

**Step 4: `DataLoader` turns examples into a batch.** Each time the training loop asks for the next batch (`for inputs, targets in loader`), the DataLoader:

1. picks `batch_size` indices: here 0, 1, 2, in order. In real training, `shuffle=True` draws a fresh random order every epoch.
2. calls `tds[0]`, `tds[1]`, `tds[2]`, which runs step 3 three times. Each call returns an `(inputs, targets)` pair of length-4 tensors.
3. **stacks** them (PyTorch's default "collate" function): the three `inputs` tensors become the rows of one 3 × 4 tensor, and the three `targets` tensors likewise.

```
tds[0] → ([0, 0, 1, 2], [0, 0, 2, 3])   ─┐
tds[1] → ([0, 4, 5, 1], [0, 5, 1, 2])    ├─ stack as rows ─→  inputs (3 × 4), targets (3 × 4)
tds[2] → ([0, 0, 6, 1], [0, 0, 1, 2])   ─┘
```

The result is the batch the model receives:

```
inputs  = [[0, 0, 1, 2],      Ann:  shampoo, conditioner
           [0, 4, 5, 1],      Ben:  red polish, top coat, shampoo
           [0, 0, 6, 1]]      Cat:  perfume, shampoo

targets = [[0, 0, 2, 3],      Ann:  predict conditioner, then hair oil
           [0, 5, 1, 2],      Ben:  predict top coat, shampoo, conditioner
           [0, 0, 1, 2]]      Cat:  predict shampoo, then conditioner
```

Real training uses the same code with `max_len = 50` and `batch_size = 256`. That gives 256 × 50 tensors, 88 batches per epoch, and the last batch is smaller: 22,363 − 87 × 256 = 91 users.

### Counting the training signals

Every **non-zero target** is one quiz, "given what you've seen so far, what comes next?". Read Ann's row column by column. Because of the causal mask (Part 2), position $`t`$ sees only the inputs up to $`t`$:

| Position | Model sees | Must predict | A signal? |
|---|---|---|---|
| 0 | PAD | 0 | ✗ padding: nothing to predict |
| 1 | PAD, PAD | 0 | ✗ |
| 2 | shampoo | 2, conditioner | ✓ |
| 3 | shampoo, conditioner | 3, hair oil | ✓ |

A training list of $`n`$ items gives $`n - 1`$ signals, because `s[1:]` has $`n - 1`$ items. Ann (3 items) gives 2, Ben (4) gives 3, and Cat (3) gives 2, for **7 training signals** in this batch. All 7 are answered in **one** forward pass. The loss in 8.7 is their average.

**Model:** the real `SASRec` class, shrunk to `hidden = 4`, 1 layer, 1 head, no dropout. It has 296 learnable numbers in total; the real one has 877,824.

---

## 8.7 The forward pass

The training loop calls `model.loss(inputs, targets)`, [sasrec.py:64-66](../src/genrec/models/sasrec.py#L64-L66):

```python
def loss(self, inputs, targets):
    logits = self.logits(self.encode(inputs))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
```

We'll go through it from the inside out.

### Step 1: embedding lookup, `x = E[inputs] + P`

[sasrec.py:50](../src/genrec/models/sasrec.py#L50): `x = self.item_emb(seqs) + self.pos_emb(positions)`.

The freshly initialized tables (row 0 of $`\mathbf{E}`$ is the zero padding vector, and all other values are small random numbers, standard deviation 0.02):

```
E (item_emb, 7 × 4)                        P (pos_emb, 4 × 4)
0 PAD          [ 0.000  0.000  0.000  0.000]   pos 0 [-0.003 -0.001  0.008  0.023]
1 shampoo      [-0.018 -0.012  0.003 -0.021]   pos 1 [ 0.039  0.020  0.020 -0.009]
2 conditioner  [-0.023 -0.006  0.019 -0.023]   pos 2 [-0.025  0.026  0.005  0.011]
3 hair oil     [-0.012 -0.004 -0.010  0.016]   pos 3 [-0.000 -0.045  0.029 -0.024]
4 red polish   [ 0.002 -0.021 -0.002 -0.020]
5 top coat     [-0.024  0.010 -0.025 -0.027]
6 perfume      [ 0.031 -0.001 -0.008 -0.005]
```

For Ann (`inputs = [0, 0, 1, 2]`), each row of $`\mathbf{x}`$ is one item row of $`\mathbf{E}`$ plus one position row of $`\mathbf{P}`$:

| Position | Item | $`\mathbf{E}[\text{item}] + \mathbf{P}[\text{pos}]`$ |
|---|---|---|
| 0 | PAD | $`(0,0,0,0) + (-0.003, -0.001, 0.008, 0.023) = (-0.003, -0.001, 0.008, 0.023)`$ |
| 1 | PAD | $`(0.039, 0.020, 0.020, -0.009)`$ |
| 2 | shampoo | $`(-0.018, -0.012, 0.003, -0.021) + (-0.025, 0.026, 0.005, 0.011) = (-0.043, 0.014, 0.008, -0.010)`$ |
| 3 | conditioner | $`(-0.023, -0.006, 0.019, -0.023) + (-0.000, -0.045, 0.029, -0.024) = (-0.023, -0.051, 0.048, -0.047)`$ |

Shape: **(3 users, 4 positions, 4 numbers)**. At full scale it's (256, 50, 64).

### Step 2: attention and normalization, `h = encode(inputs)`

[sasrec.py:54-59](../src/genrec/models/sasrec.py#L54-L59) builds the mask (no looking at future or padded positions, Part 2) and runs the Transformer layer and a final layer norm. Ann's output:

```
h[Ann] = pos 0  [-0.719  1.100 -1.239  0.857]
         pos 1  [-0.680  1.165  0.773 -1.258]
         pos 2  [ 0.102  0.241  1.219 -1.562]     ← summary of (shampoo)
         pos 3  [ 0.284  0.009  1.245 -1.538]     ← summary of (shampoo, conditioner)
```

These numbers are much larger than $`\mathbf{x}`$, because layer norm rescales each row to mean 0 and variance 1. Row 3 is Ann's $`\mathbf{u}_h`$ from Part 0: the model's summary of "shampoo, then conditioner", used to predict what comes next.

### Step 3: score every item, `logits = h @ E.T`

[sasrec.py:61-62](../src/genrec/models/sasrec.py#L61-L62): a dot product of every position's $`\mathbf{h}`$ with every row of $`\mathbf{E}`$. Shape **(3, 4, 7)**: for every user and position, one score per item, padding included.

Ann, position 3 (target: **3, hair oil**):

| Item | PAD | shampoo | conditioner | **hair oil** | red polish | top coat | perfume |
|---|---|---|---|---|---|---|---|
| logit | 0.000 | 0.031 | 0.052 | **−0.040** | 0.029 | 0.005 | 0.006 |
| softmax | 0.141 | 0.146 | 0.149 | **0.136** | 0.145 | 0.142 | 0.142 |

The untrained model has **no idea**: every probability is close to $`1/7 = 0.143`$. Item vectors are tiny, so all the dot products are near 0.

### Step 4: the loss

Cross-entropy for this one position, by hand: $`-\log 0.1356 = 1.998`$. The real code computes this for every position at once:

```
per-position loss    pos 0   pos 1   pos 2   pos 3
Ann                  0       0       1.904   1.998
Ben                  0       1.913   1.929   1.899
Cat                  0       0       1.946   1.903
```

`ignore_index=0` gives 0 for positions whose target is padding and **leaves them out of the average**. The loss is the mean over the **7** real targets:

$$
\mathcal{L} = \frac{1.904 + 1.998 + 1.913 + 1.929 + 1.899 + 1.946 + 1.903}{7} = 1.927
$$

Compare with the loss of a uniform guess over 7 classes, $`\ln 7 = 1.946`$. Before training, the model is no better than random, as expected.

---

## 8.8 The backward pass

Back in the training loop, [train.py](../scripts/train.py):

```python
loss = model.loss(inputs.to(device), targets.to(device))
opt.zero_grad()      # clear gradients left over from the previous batch
loss.backward()      # compute d(loss)/d(weight) for every one of the 296 weights
opt.step()           # update every weight (next section)
```

### The gradient at the logits

`loss.backward()` applies the chain rule through every operation, in reverse. Its first step has a clean formula. For softmax with cross-entropy, the gradient of the loss with respect to the logits is

$$
\frac{\partial \mathcal{L}}{\partial \text{logits}} = \frac{\text{softmax} - \text{onehot}(\text{target})}{\text{number of real targets}}
$$

For Ann's position 3 (target = hair oil, divided by 7):

| | PAD | shampoo | conditioner | **hair oil** | red polish | top coat | perfume |
|---|---|---|---|---|---|---|---|
| softmax | 0.141 | 0.146 | 0.149 | **0.136** | 0.145 | 0.142 | 0.142 |
| − onehot | 0 | 0 | 0 | **−1** | 0 | 0 | 0 |
| ÷ 7 | **+0.020** | **+0.021** | **+0.021** | **−0.123** | **+0.021** | **+0.020** | **+0.020** |

PyTorch's computed gradient, read back from the real run, is exactly this row. In words: **push the true item's score up, and push every other item's score down a little, in proportion to how much probability it wrongly got.**

### Where it flows next

From the logits, the gradient flows back into $`\mathbf{h}`$, then through layer norm and attention into their weights, then into $`\mathbf{x}`$, and finally into the rows of $`\mathbf{E}`$ and $`\mathbf{P}`$ that were looked up.

One thing is surprising at first. **Every** row of $`\mathbf{E}`$ gets a gradient, even items that appear nowhere in the batch, and even the PAD row:

```
gradient size per item row:  PAD 0.254 | shampoo 0.435 | conditioner 0.577 | hair oil 0.175 | red polish 0.340 | top coat 0.045 | perfume 0.252
```

That's because $`\mathbf{E}`$ is also the output layer (logits = $`\mathbf{h} \cdot \mathbf{E}^\top`$), and every item is in every softmax denominator. This is the **cost of a full softmax**, from Part 0: at real scale every step touches all 12,102 rows. The logits tensor alone is 256 × 50 × 12,102 ≈ 155 million numbers, about 620 MB.

---

## 8.9 The update: one Adam step

`opt.step()` updates every weight using its gradient. Plain gradient descent would do $`w \leftarrow w - \text{lr} \cdot g`$. We use **Adam**, which keeps two running averages per weight: one of its gradient $`m`$, and one of its squared gradient $`v`$. It then steps by

$$
w \leftarrow w - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

where $`\hat{m}`$ and $`\hat{v}`$ are those averages, corrected for starting at zero. On the **very first step**, $`\hat{m} = g`$ and $`\hat{v} = g^2`$, so the step is $`\text{lr} \cdot g / |g| = \pm\,\text{lr}`$: every weight moves by about the learning rate, in the direction that lowers the loss, however big or small its gradient was.

From the real run (lr = 0.001):

| | Value |
|---|---|
| One attention weight, before | −0.5707 |
| Its gradient | −0.0000142 (tiny) |
| After | −0.5697 (moved **+0.0010**) |
| Weights whose first step was ≈ 0.001 | 88% |
| Loss on the same batch, before → after | 1.927 → **1.922** |

The gradient was tiny, but Adam still moved the weight by a full 0.001. That's Adam's point: each weight gets its own step size, so weights with tiny gradients still learn.

### What repeating this does

Repeat the step on the **same** batch, just to watch learning happen:

| Steps | 1 | 10 | 50 | 200 |
|---|---|---|---|---|
| Loss | 1.922 | 1.885 | 1.710 | 0.970 |
| P(hair oil) for Ann's position 3 | | | | 0.342 (started at 0.136) |

The model is memorizing these 3 users. Real training never repeats one batch: each of the 88 batches per epoch holds different users, so the weights must find patterns that hold across users. Part 2 showed that this generalization stops improving around epoch 30.

---

## 8.10 The loop around it

All of this happens inside [train.py](../scripts/train.py) `fit_with_early_stopping`:

```python
opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
for epoch in range(1, cfg["epochs"] + 1):            # up to 50 epochs
    model.train()                                    # turn dropout on
    for inputs, targets in loader:                   # 88 batches, new random order each epoch
        loss = model.loss(inputs.to(device), targets.to(device))     # 8.7
        opt.zero_grad()
        loss.backward()                                              # 8.8
        opt.step()                                                   # 8.9
    if epoch % cfg["eval_every"]:                    # every 5 epochs...
        continue
    metrics = evaluate(model.recommend(valid[0], k), valid[1], cfg["eval_ks"])   # ...check validation (Part 1)
    # keep the best weights by valid NDCG@10; stop after 4 checks without improvement
```

| Level | Count, full SASRec run |
|---|---|
| Steps (batches) per epoch | 88 |
| Epochs | 50 |
| Total training steps | 4,400 |
| Validation checks | 10 (every 5 epochs) |
| Test evaluations | 1 (at the very end, with the best weights) |

---

## 8.11 The same walk for TIGER

TIGER (Part 6) uses the same loop, `fit_with_early_stopping`, but prepares its data differently. Using the same toy users, now with hand-made 3-code Semantic IDs ($`K = 4`$):

| Item | Semantic ID | Tokens: $`1 + l \cdot 4 + c`$ |
|---|---|---|
| shampoo | (0, 0, 0) | 1, 5, 9 |
| conditioner | (0, 1, 0) | 1, 6, 9 |
| hair oil | (0, 2, 0) | 1, 7, 9 |
| red polish | (1, 0, 0) | 2, 5, 9 |
| top coat | (1, 1, 0) | 2, 6, 9 |
| perfume | (2, 0, 0) | 3, 5, 9 |

BOS is token 13. In the real model this table is the `item_tokens` buffer.

**Examples:** `NextItemPairs` makes one example per position (Part 6). Ann's training list (1, 2, 3) gives two:

| Example | History (left-padded, `max_items = 3`) | Target |
|---|---|---|
| 1 | [0, 0, 1] (shampoo) | 2, conditioner |
| 2 | [0, 1, 2] (shampoo, conditioner) | 3, hair oil |

**Encoder input:** each history item becomes its 3 tokens:

```
example 1: [0 0 0 | 0 0 0 | 1 5 9]
example 2: [0 0 0 | 1 5 9 | 1 6 9]
```

The first item column is padding for **both** rows, so `encode` trims it ([tiger.py](../src/genrec/models/tiger.py), Part 6's speed fix). The encoder memory has 6 token positions, not 9.

**Decoder input and targets** (teacher forcing):

| Example | Target codes | Decoder input tokens | Must predict |
|---|---|---|---|
| 1 | (0, 1, 0) | [13, 1, 6] = [BOS, code 0 at level 1, code 1 at level 2] | 0, then 1, then 0 |
| 2 | (0, 2, 0) | [13, 1, 7] | 0, then 2, then 0 |

**Loss** from the real `TIGER.loss`: the average over the 3 levels of the cross-entropy over 4 codes:

| Level | 1 | 2 | 3 | Mean |
|---|---|---|---|---|
| Loss | 0.460 | 1.254 | 1.373 | **1.029** |

Compare with $`\ln 4 = 1.386`$ for a uniform guess over 4 codes. The level-1 value happens to start lower only because of the random initial weights. From here, `backward` and `opt.step` work exactly as in 8.8–8.9.

### SASRec vs. TIGER data preparation

| | SASRec | TIGER |
|---|---|---|
| One example is | one user's whole training list, shifted by one | one (history prefix, next item) pair |
| Examples per epoch | 22,363 (≈ 128k signals inside them) | 131,413 |
| Model input | item indices | Semantic ID tokens, 4 per item |
| Prediction target | 1 item out of 12,101 | 4 codes, each out of 256 |
| Batching | random | grouped by history length (`BucketBatchSampler`) |

---

## 8.12 Check yourself

<details>
<summary><b>Q1.</b> A user's training list is (7, 3, 9, 4). With max_len = 4, what are the inputs and targets, and how many training signals are there?</summary>

`s = [7, 3, 9, 4]`; inputs = `left_pad([7, 3, 9], 4)` = `[0, 7, 3, 9]`; targets = `left_pad([3, 9, 4], 4)` = `[0, 3, 9, 4]`. That's 3 training signals: 7 → 3, (7, 3) → 9, (7, 3, 9) → 4.
</details>

<details>
<summary><b>Q2.</b> In 8.7, why is the batch loss divided by 7 and not by 12 (3 users × 4 positions)?</summary>

Five positions have target 0 (padding), and `ignore_index=0` leaves them out entirely. Averaging over them would dilute the loss with meaningless positions, and users with short histories would count less.
</details>

<details>
<summary><b>Q3.</b> Padding positions are excluded from the loss, and the embedding layer is told to ignore index 0 (padding_idx=0). Why does the PAD row of E still get a gradient (0.254)?</summary>

`padding_idx=0` only blocks the gradient coming from the **input lookup**. But $`\mathbf{E}`$ is also the output layer: the logits are $`\mathbf{h} \cdot \mathbf{E}^\top`$, so PAD has a score in every softmax, and its row gets the same "push down a little" gradient as every other wrong item. It's harmless: padding positions are masked out of attention, and `recommend` sets PAD's score to $`-\infty`$ before picking items.
</details>

<details>
<summary><b>Q4.</b> Adam moved a weight with gradient −0.0000142 by a full +0.001 on the first step. Would plain gradient descent with lr = 0.001 do the same?</summary>

No. It would move it by $`0.001 \times 0.0000142 \approx 0.00000001`$, which is essentially nothing. Adam divides by the square root of the running average of squared gradients, which normalizes each weight's step size.
</details>

<details>
<summary><b>Q5.</b> How might the same-day ties in 8.2 make our test scores look better than real-world performance?</summary>

For users with many same-day items, the "next" item is often just the next product ID in alphabetical ASIN order, and products from the same line often have adjacent IDs. A model can learn "after item 9194 comes 9195" from that artifact. Real users don't buy in ASIN order, so a deployed model couldn't rely on it.
</details>

---

**Back to:** [the contents](README.md).
