import numpy as np


def detect_instable_ghosts(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray,
    ghost_indices: np.ndarray = None,
):
    """
    Calculate the variance of the original and ghost points within embeddings.

    Parameters
    ----------
    original_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_components)
    ghost_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_ghosts, n_components)
    ghost_indices: np.ndarray of shape (n_ghost_targets,)

    Returns
    -------
    rank: np.ndarray of shape (n_ghost_targets,)
        The rank of the ghosts based on the variance.
    score: np.ndarray of shape (n_ghost_targets,)
        The variance of the ghosts.
    """
    if ghost_indices is None:
        ghost_indices = np.arange(original_embeddings.shape[1])

    original_embeddings = original_embeddings[:, ghost_indices]
    ghost_embeddings = ghost_embeddings[:, ghost_indices]

    E = original_embeddings[:, :, np.newaxis]
    E = np.concatenate([E, ghost_embeddings], axis=2)
    # shape of E: (n_embeddings, n_samples, n_ghosts+1, n_components)

    Mu = np.mean(E, axis=2)  # shape of M: (n_embeddings, n_samples, n_components)

    W = np.sum(np.square(E - Mu[:, :, np.newaxis]), axis=3)
    W = np.mean(W, axis=(2, 0))

    rank = np.argsort(W)[::-1]
    score = W[rank]
    if ghost_indices is not None:
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


# def calculate_variances(
#     original_embeddings: np.ndarray,
#     ghost_embeddings: np.ndarray = None,
#     ghost_indices: np.ndarray = None,
# ):

#     W = within_embedding_variance(original_embeddings, ghost_embeddings, ghost_indices)
#     B = between_embedding_variance(original_embeddings, ghost_embeddings, ghost_indices)

#     V = W + B

#     rank = np.argsort(V)[::-1]
#     score = V[rank]
#     if ghost_indices is not None:
#         rank = ghost_indices[rank]

#     return rank, score


# def within_embedding_variance(
#     original_embeddings: np.ndarray,
#     ghost_embeddings: np.ndarray = None,
#     ghost_indices: np.ndarray = None,
# ):
#     """
#     Calculate the variance of the original and ghost points within embeddings.

#     Parameters
#     ----------
#     original_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_components)
#     ghost_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_ghosts, n_components)
#     ghost_indices: np.ndarray of shape (n_ghost_targets,)

#     Returns
#     -------
#     WV: np.ndarray of shape (n_ghost_targets,)
#         Mean within-embedding variance for each ghost target.
#     """
#     if ghost_indices is None:
#         ghost_indices = np.arange(original_embeddings.shape[1])
#     if ghost_embeddings is None:
#         return np.zeros(len(ghost_indices))

#     original_embeddings = original_embeddings[:, ghost_indices]
#     ghost_embeddings = ghost_embeddings[:, ghost_indices]

#     E = original_embeddings[:, :, np.newaxis]
#     if ghost_embeddings.shape[2]:
#         E = np.concatenate([E, ghost_embeddings], axis=2)
#     # shape of E: (n_embeddings, n_samples, n_ghosts+1, n_components)

#     Mu = np.mean(E, axis=2)  # shape of M: (n_embeddings, n_samples, n_components)

#     W = np.sum(np.square(E - Mu[:, :, np.newaxis]), axis=3)
#     W = np.mean(W, axis=(2, 0))

#     return W
