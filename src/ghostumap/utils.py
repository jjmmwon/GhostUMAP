from typing import Literal
import numba
import numpy as np


def calculate_variances(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray,
    ghost_indices: np.ndarray,
    knn_indices: np.ndarray = None,
    metric: Literal["absolute", "relative"] = "absolute",
):
    if metric == "absolute":
        return _calculate_abs_variances(
            original_embeddings, ghost_embeddings, ghost_indices
        )
    elif metric == "relative":
        if knn_indices is None:
            raise ValueError(
                "knn_indices must be provided for relative variance calculation."
            )
        return _calculate_rel_variances(
            original_embeddings, ghost_embeddings, ghost_indices, knn_indices
        )
    else:
        raise ValueError(f"Unknown metric {metric}")


def _calculate_abs_variances(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray = None,
    ghost_indices: np.ndarray = None,
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
    if ghost_embeddings is None:
        ghost_embeddings = np.empty((0, 0, 0, original_embeddings.shape[-1]))
    if ghost_indices is None:
        ghost_indices = list(range(original_embeddings.shape[1]))

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


def _calculate_rel_variances(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray = None,
    ghost_indices: np.ndarray = None,
    knn_indices: np.ndarray = None,
):
    """
    Calculate the relative variance of the original and ghost embeddings.
    The relative variance is calculated as the variance of the position of each point in the embedding space

    Parameters
    ----------
    original_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_components)
        The original embeddings.
    ghost_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_ghosts, n_components)
        The ghost embeddings.
    ghost_indices: np.ndarray of shape (n_ghost_targets,)
        The indices of the ghost embeddings.
    knn_indices: np.ndarray of shape (n_samples, n_neighbors)
        The indices of the nearest neighbors for each point in the original embeddings.

    Returns
    -------
    ranks: np.ndarray of shape (n_ghost_targets,)
        The relative rank of the original embeddings.
    variances: np.ndarray of shape (n_ghost_targets,)
        The relative variance of the original embeddings.
    """
    if knn_indices is None:
        raise ValueError(
            "knn_indices must be provided for relative variance calculation."
        )
    if ghost_embeddings is None:
        ghost_embeddings = np.empty((0, 0, 0, original_embeddings.shape[-1]))
    if ghost_indices is None:
        ghost_indices = list(range(original_embeddings.shape[1]))

    total_points = original_embeddings[:, ghost_indices, np.newaxis]
    if ghost_embeddings.shape[2]:
        total_points = np.concatenate(
            [total_points, ghost_embeddings[:, ghost_indices]], axis=2
        )

    total_points = np.swapaxes(total_points, 0, 1)
    # shape of total_points: (n_ghost_targets, n_embeddings, n_ghosts+1, n_components)

    knn_coords = original_embeddings[:, ghost_indices]
    knn_coords = knn_coords[:, knn_indices[ghost_indices, 1:]]
    knn_coords = np.swapaxes(knn_coords, 0, 1)
    # shape of knn_coords: (n_ghost_targets, n_embeddings, n_neighbors, n_components)

    distances = np.linalg.norm(
        knn_coords[:, :, :, np.newaxis] - total_points[:, :, np.newaxis], axis=4
    )  # shape: (n_ghost_targets, n_embeddings, n_neighbors, n_ghosts+1)
    # distances matrix per target point in each embedding

    n_ghost_targets, n_embeddings, n_neighbors, n_ghosts = distances.shape
    distances = np.swapaxes(distances, 1, 2).reshape(n_ghost_targets, n_neighbors, -1)
    # shape: (n_ghost_targets, n_neighbors, n_embeddings * n_ghosts)

    var_dists = np.var(distances, axis=1)
    mean_var_dists = np.mean(var_dists, axis=1)

    ranks = np.argsort(mean_var_dists)[::-1]
    variances = mean_var_dists[ranks]

    return ranks, variances
