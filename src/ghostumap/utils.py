import numpy as np


def measure_instability(
    original_embedding: np.ndarray,
    ghost_embeddings: np.ndarray,
    ghost_indices: np.ndarray = None,
):
    """
    Calculate the variance of the original and ghost points within embeddings.

    Parameters
    ----------
    original_embeddings: np.ndarray of shape (n_samples, n_components)
    ghost_embeddings: np.ndarray of shape (n_samples, n_ghosts, n_components)
    ghost_indices: np.ndarray of shape (n_ghost_targets,)

    Returns
    -------
    rank: np.ndarray of shape (n_ghost_targets,)
        The rank of the ghosts based on the variance.
    score: np.ndarray of shape (n_ghost_targets,)
        The variance of the ghosts.
    """
    if ghost_indices is None:
        ghost_indices = np.arange(original_embedding.shape[0])

    O = original_embedding[ghost_indices, :]
    G = ghost_embeddings[ghost_indices, :, :]

    # Y = O[:, np.newaxis]
    Y = np.concatenate([O[:, np.newaxis], G], axis=1)
    # shape of Y: (n_samples, n_ghosts+1, n_components)

    Mu = np.mean(Y, axis=1)  # shape of Mu: (n_samples, n_components)

    INS = np.sum(np.square(Y - Mu[:, np.newaxis]), axis=2)
    INS = np.mean(INS, axis=1)
    # shape of INS: (n_samples,)

    rank = np.argsort(INS)[::-1]
    score = INS[rank]

    rank = ghost_indices[rank]

    return rank, score


# def detect_instable_ghosts(
#     original_embedding: np.ndarray,
#     ghost_embeddings: np.ndarray,
#     ghost_indices: np.ndarray = None,
# ):
#     """
#     Parameters
#     ----------
#     original_embeddings: np.ndarray of shape (n_samples, n_components)
#     ghost_embeddings: np.ndarray of shape (n_samples, n_ghosts, n_components)
#     ghost_indices: np.ndarray of shape (n_ghost_targets,)

#     Returns
#     -------


#     """
#     if ghost_indices is None:
#         ghost_indices = np.arange(original_embedding.shape[0])

#     O = original_embedding[ghost_indices]
#     G = ghost_embeddings[ghost_indices]
#     Y = np.concatenate([O[:, np.newaxis], G], axis=1)
#     Mu = np.mean(Y, axis=1)

#     INS = np.sum(np.square(Y - Mu[:, np.newaxis]), axis=2)
#     INS = np.mean(INS, axis=1)

#     rank = np.argsort(INS)[::-1]
#     var = INS[rank]
#     rank = ghost_indices[rank]

#     return rank, var
