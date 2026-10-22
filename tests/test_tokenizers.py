import numpy as np

from genrec.tokenizers.kmeans import add_collision_suffix, code_stats, hierarchical_kmeans, residual_kmeans


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
