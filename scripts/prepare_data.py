"""Download and preprocess an Amazon 2014 category into data/processed/<name>/."""

import argparse

from genrec.data.amazon import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--category", default="Beauty")
parser.add_argument("--raw-dir", default="data/raw")
parser.add_argument("--out-dir", default="data/processed/beauty")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

prepare(args.category, args.raw_dir, args.out_dir, args.k)
print(f"Wrote {args.out_dir}/sequences.json and items.csv")
