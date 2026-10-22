"""Train and evaluate a model from a YAML config.

Writes experiments/<name>-<timestamp>/{config.yaml, metrics.json}.

  python scripts/train.py --config configs/sasrec_beauty.yaml
"""

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from genrec.data.amazon import load_processed
from genrec.data.sequences import NextItemDataset, leave_one_out
from genrec.metrics import evaluate
from genrec.models.baselines import Popularity
from genrec.models.sasrec import SASRec
from genrec.utils import get_device, load_config, set_seed


def run_popularity(cfg, train, valid, test):
    model = Popularity().fit(train)
    k = max(cfg["eval_ks"])
    return {
        "valid": evaluate(model.recommend(valid[0], k), valid[1], cfg["eval_ks"]),
        "test": evaluate(model.recommend(test[0], k), test[1], cfg["eval_ks"]),
    }


def run_sasrec(cfg, train, valid, test, n_items):
    device = get_device()
    model = SASRec(
        n_items, cfg["max_len"], cfg["hidden"], cfg["n_layers"], cfg["n_heads"], cfg["dropout"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(NextItemDataset(train, cfg["max_len"]), batch_size=cfg["batch_size"], shuffle=True)
    k = max(cfg["eval_ks"])

    best_ndcg, best_state, bad_evals = -1.0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start, total = time.time(), 0.0
        for inputs, targets in loader:
            loss = model.loss(inputs.to(device), targets.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % cfg["eval_every"]:
            continue

        metrics = evaluate(model.recommend(valid[0], k), valid[1], cfg["eval_ks"])
        print(
            f"epoch {epoch:3d}  loss {total / len(loader):.4f}  "
            f"valid recall@10 {metrics['recall@10']:.4f}  ndcg@10 {metrics['ndcg@10']:.4f}  "
            f"({time.time() - start:.1f}s/epoch)"
        )
        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg, best_state, bad_evals = metrics["ndcg@10"], copy.deepcopy(model.state_dict()), 0
            best_valid, best_epoch = metrics, epoch
        else:
            bad_evals += 1
            if bad_evals >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "valid": best_valid,
        "test": evaluate(model.recommend(test[0], k), test[1], cfg["eval_ks"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg.get("seed", 42))

    sequences, items = load_processed(cfg["data_dir"])
    train, valid, test = leave_one_out(sequences)

    if cfg["model"] == "popularity":
        results = run_popularity(cfg, train, valid, test)
    elif cfg["model"] == "sasrec":
        results = run_sasrec(cfg, train, valid, test, n_items=len(items))
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    run_dir = Path("experiments") / f"{cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
