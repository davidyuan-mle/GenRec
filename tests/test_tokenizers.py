import numpy as np
import torch

from genrec.tokenizers.kmeans import add_collision_suffix, code_stats, hierarchical_kmeans, residual_kmeans
from genrec.tokenizers.rqvae import RQVAE, ResidualQuantizer, fit_rqvae


def test_add_collision_suffix_numbers_duplicates():
    codes = np.array([[1, 2], [1, 2], [3, 4], [1, 2]])
    assert add_collision_suffix(codes)[:, -1].tolist() == [0, 1, 0, 2]


def test_code_stats_collisions():
    stats = code_stats(np.array([[1, 2], [1, 2], [3, 4]]))
    assert stats["unique_ids"] == 2
    assert stats["collision_rate"] == 2 / 3
    assert stats["max_group"] == 2


def test_residual_kmeans_refines_reconstruction():
    emb = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    _, recon1 = residual_kmeans(emb, k=4, n_levels=1)
    codes, recon3 = residual_kmeans(emb, k=4, n_levels=3)
    assert codes.shape == (200, 3) and codes.max() < 4
    assert np.linalg.norm(emb - recon3) < np.linalg.norm(emb - recon1)


def test_hierarchical_kmeans_codes_are_unique_when_leaves_fit():
    # 4 well-separated groups of 3: level 1 finds the groups, and each group (3 <= k)
    # gives every item its own level-2 code.
    rng = np.random.default_rng(0)
    emb = (np.repeat(10 * np.eye(4), 3, axis=0) + rng.normal(scale=0.1, size=(12, 4))).astype(np.float32)
    codes, recon = hierarchical_kmeans(emb, k=4, n_levels=2)
    assert len(set(codes[:, 0])) == 4
    assert len(set(map(tuple, codes))) == 12
    assert np.allclose(recon, emb)


def test_rqvae_straight_through_reaches_encoder():
    torch.manual_seed(0)
    model = RQVAE(in_dim=8, hidden=[16], latent=4, k=4, n_levels=2)
    _, codes, recon_loss, q_loss = model(torch.randn(32, 8))
    (recon_loss + q_loss).backward()
    assert codes.shape == (32, 2)
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert all(cb.weight.grad is not None for cb in model.quantizer.codebooks)


def test_reset_dead_codes_revives_unused_codes():
    torch.manual_seed(0)
    q = ResidualQuantizer(n_levels=1, k=4, dim=2)
    z = torch.randn(64, 2)
    with torch.no_grad():
        q.codebooks[0].weight[1:] = 100.0  # far from every point: codes 1-3 are dead
    assert q.reset_dead_codes(z) == 3
    assert q.codebooks[0].weight.abs().max() < 100.0


def test_fit_rqvae_shapes():
    emb = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
    cfg = {"hidden": [16], "latent": 4, "k": 4, "levels": 2, "beta": 0.25, "lr": 1e-2,
           "batch_size": 32, "epochs": 2, "log_every": 1}
    codes, recon, history = fit_rqvae(emb, cfg, log=lambda _: None)
    assert codes.shape == (64, 2) and recon.shape == emb.shape and len(history) == 2
