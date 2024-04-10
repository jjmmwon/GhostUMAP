import numpy as np


def calculate_variances(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray,
    ghost_indices: np.ndarray,
):
    """
    Calculate the variance of the original and ghost embeddings.
    The variance is calculated as the variance of the position of each point in the embedding space.

    Parameters
    ----------
    original_embeddings: np.ndarray of shape (n_embeddings, n_vertices, n_components)
        The original embeddings.
    ghost_embeddings: np.ndarray of shape (n_embeddings, n_vertices, n_ghosts_per_target, n_components)
        The ghost embeddings.
    ghost_indices: np.ndarray of shape (n_ghosts,)
        The indices of the ghost embeddings.

    Returns
    -------
    rank: np.ndarray of shape (n_ghosts,)
        The rank of the variance of the original embeddings.
    variances: np.ndarray of shape (n_ghosts,)
        The variance of the original embeddings.

    """

    total_points = np.swapaxes(original_embeddings[:, ghost_indices], 0, 1)
    if ghost_embeddings.shape[2]:
        ghost_points = np.swapaxes(ghost_embeddings[:, ghost_indices], 0, 1).reshape(
            ghost_embeddings.shape[1], -1, ghost_embeddings.shape[-1]
        )
        total_points = np.concatenate([total_points, ghost_points], axis=1)

    mean = np.mean(total_points, axis=1)
    dists = np.linalg.norm(total_points - mean[:, np.newaxis], axis=2)

    var = np.var(dists, axis=1)
    rank = np.argsort(var)[::-1]
    var = var[rank]

    return rank, var
