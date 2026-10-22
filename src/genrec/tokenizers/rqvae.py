"""RQ-VAE Semantic IDs (TIGER, Rajput et al. 2023; the basis of PLUM's tokenizer).

An MLP encoder maps an item embedding to a small latent z. A residual quantizer
replaces z with a sum of codebook vectors, one per level: level l picks the code
nearest to what levels < l left unexplained. A decoder reconstructs the embedding
from the quantized latent. The chosen code indices are the Semantic ID.

Loss = reconstruction MSE + sum over levels of
       ||sg[r_l] - e_l||^2 (codebook) + beta * ||r_l - sg[e_l]||^2 (commitment),
with a straight-through estimator so gradients reach the encoder.

Two standard fixes against codebook collapse: codebooks start from k-means on
the first latents, and codes no item uses are periodically reset to random residuals.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


def mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])  # no activation on the output


class ResidualQuantizer(nn.Module):
    def __init__(self, n_levels: int, k: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebooks = nn.ModuleList(nn.Embedding(k, dim) for _ in range(n_levels))
        self.beta = beta

    def forward(self, z: torch.Tensor):
        """Return (straight-through quantized z, codes (n, L), quantization loss)."""
        residual, z_q, codes, loss = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:
            idx = torch.cdist(residual, cb.weight).argmin(1)
            e = cb(idx)
            loss = loss + F.mse_loss(e, residual.detach()) + self.beta * F.mse_loss(residual, e.detach())
            z_q = z_q + e
            residual = residual - e.detach()
            codes.append(idx)
        return z + (z_q - z).detach(), torch.stack(codes, 1), loss

    @torch.no_grad()
    def residuals(self, z: torch.Tensor):
        """Yield (level, residual entering that level, codes chosen there)."""
        residual = z
        for level, cb in enumerate(self.codebooks):
            idx = torch.cdist(residual, cb.weight).argmin(1)
            yield level, residual, idx
            residual = residual - cb(idx)

    @torch.no_grad()
    def init_kmeans(self, z: torch.Tensor, seed: int = 42):
        residual = z
        for level, cb in enumerate(self.codebooks):
            km = KMeans(cb.num_embeddings, random_state=seed + level).fit(residual.cpu().numpy())
            cb.weight.copy_(torch.from_numpy(km.cluster_centers_).to(cb.weight))
            residual = residual - cb.weight[torch.from_numpy(km.labels_).to(z.device)]

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor) -> int:
        """Point every unused code at a random residual from its level; return how many were reset."""
        n_reset = 0
        for level, residual, idx in self.residuals(z):
            cb = self.codebooks[level]
            dead = torch.bincount(idx, minlength=cb.num_embeddings) == 0
            if dead.any():
                pick = torch.randint(len(residual), (int(dead.sum()),), device=z.device)
                cb.weight[dead] = residual[pick]
                n_reset += int(dead.sum())
        return n_reset


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], latent: int, k: int, n_levels: int, beta: float = 0.25):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.quantizer = ResidualQuantizer(n_levels, k, latent, beta)

    def forward(self, x: torch.Tensor):
        """Return (reconstruction, codes, reconstruction loss, quantization loss)."""
        z_q, codes, q_loss = self.quantizer(self.encoder(x))
        x_hat = self.decoder(z_q)
        return x_hat, codes, F.mse_loss(x_hat, x), q_loss


def fit_rqvae(emb: np.ndarray, cfg: dict, device: str = "cpu", log=print):
    """Train an RQ-VAE on item embeddings (one row per item, no padding row).

    Returns (codes (n, L), recon (n, dim) in the original embedding space, history).
    Inputs are standardized per dimension; recon is mapped back.
    """
    torch.manual_seed(cfg.get("seed", 42))
    mean, std = emb.mean(0), emb.std(0) + 1e-8
    x = torch.from_numpy((emb - mean) / std).float().to(device)

    model = RQVAE(x.shape[1], cfg["hidden"], cfg["latent"], cfg["k"], cfg["levels"], cfg["beta"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    with torch.no_grad():
        model.quantizer.init_kmeans(model.encoder(x), cfg.get("seed", 42))

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for batch in torch.randperm(len(x), device=device).split(cfg["batch_size"]):
            _, _, recon_loss, q_loss = model(x[batch])
            opt.zero_grad()
            (recon_loss + q_loss).backward()
            opt.step()

        reset_every, reset_until = cfg.get("reset_every", 0), cfg.get("reset_until", 0)
        n_reset = 0
        if reset_every and epoch % reset_every == 0 and epoch <= reset_until:
            model.eval()
            n_reset = model.quantizer.reset_dead_codes(model.encoder(x))

        if epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"]:
            model.eval()
            with torch.no_grad():
                _, codes, recon_loss, q_loss = model(x)
            codes = codes.cpu().numpy()
            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "q_loss": q_loss.item(),
                "unique_ids": len(np.unique(codes, axis=0)),
                **{f"l{i + 1}_used": len(np.unique(codes[:, i])) for i in range(codes.shape[1])},
                "reset": n_reset,
            }
            history.append(row)
            log("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

    model.eval()
    with torch.no_grad():
        x_hat, codes, _, _ = model(x)
    recon = x_hat.cpu().numpy() * std + mean
    return codes.cpu().numpy(), recon.astype(np.float32), history
