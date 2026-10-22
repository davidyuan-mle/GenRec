"""Amazon Review Data (2014) loader, e.g. the Beauty category.

Pipeline: download -> load reviews -> k-core filter -> per-user time-ordered
item sequences -> item metadata + text. Item indices start at 1; 0 is padding.

Source: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
"""

import ast
import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def raw_filenames(category: str) -> tuple[str, str]:
    return f"reviews_{category}_5.json.gz", f"meta_{category}.json.gz"


def download(category: str, raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in raw_filenames(category):
        path = raw_dir / name
        if path.exists():
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Return one row per review: user, item (ASIN), timestamp."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], r["unixReviewTime"]))
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


def load_meta(path: str | Path, asins: set[str]) -> pd.DataFrame:
    """Return metadata for the given ASINs. Lines are Python dict literals, not JSON."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            m = ast.literal_eval(line)
            if m["asin"] not in asins:
                continue
            categories = m.get("categories") or []
            rows.append(
                {
                    "asin": m["asin"],
                    "title": m.get("title", ""),
                    "brand": m.get("brand", ""),
                    "categories": " | ".join(", ".join(c) for c in categories),
                    "price": m.get("price"),
                    "description": m.get("description", ""),
                }
            )
    return pd.DataFrame(rows)


def kcore_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Repeatedly drop users and items with fewer than k interactions."""
    while True:
        user_counts = df["user"].map(df["user"].value_counts())
        item_counts = df["item"].map(df["item"].value_counts())
        keep = (user_counts >= k) & (item_counts >= k)
        if keep.all():
            return df.reset_index(drop=True)
        df = df[keep]


def item_text(row: pd.Series) -> str:
    """Text used for semantic embeddings (TIGER-style: title, brand, categories, price)."""
    parts = []
    if row["title"]:
        parts.append(f"Title: {row['title']}")
    if row["brand"]:
        parts.append(f"Brand: {row['brand']}")
    if row["categories"]:
        parts.append(f"Categories: {row['categories']}")
    if pd.notna(row["price"]):
        parts.append(f"Price: ${row['price']:.2f}")
    return ". ".join(parts)


def build_dataset(reviews: pd.DataFrame, meta: pd.DataFrame, k: int = 5):
    """Return (sequences, items).

    sequences: list of per-user item-index lists, oldest first.
    items: DataFrame indexed by item index (1..N) with asin, metadata, and text.
    """
    df = kcore_filter(reviews, k)
    # Timestamps have day resolution; a stable sort keeps file order for ties.
    df = df.sort_values("timestamp", kind="stable")

    asins = sorted(df["item"].unique())
    item_index = {a: i + 1 for i, a in enumerate(asins)}
    df["item_idx"] = df["item"].map(item_index)
    sequences = df.groupby("user", sort=True)["item_idx"].apply(list).tolist()

    items = pd.DataFrame({"asin": asins}, index=pd.RangeIndex(1, len(asins) + 1, name="item_idx"))
    items = items.merge(meta, on="asin", how="left").set_index(items.index)
    items[["title", "brand", "categories", "description"]] = (
        items[["title", "brand", "categories", "description"]].fillna("")
    )
    items["text"] = items.apply(item_text, axis=1)
    return sequences, items


def prepare(category: str, raw_dir: str | Path, out_dir: str | Path, k: int = 5) -> None:
    """Download, process, and write sequences.json + items.csv to out_dir."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    download(category, raw_dir)
    reviews_file, meta_file = raw_filenames(category)
    reviews = load_reviews(raw_dir / reviews_file)
    meta = load_meta(raw_dir / meta_file, set(reviews["item"]))
    sequences, items = build_dataset(reviews, meta, k)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sequences.json", "w") as f:
        json.dump(sequences, f)
    items.to_csv(out_dir / "items.csv")


def load_processed(out_dir: str | Path):
    """Return (sequences, items) written by prepare()."""
    out_dir = Path(out_dir)
    with open(out_dir / "sequences.json") as f:
        sequences = json.load(f)
    items = pd.read_csv(out_dir / "items.csv", index_col="item_idx", keep_default_na=False)
    return sequences, items
