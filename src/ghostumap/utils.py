import numpy as np


def calculate_variances(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray = None,
    ghost_indices: np.ndarray = None,
):

    W = within_embedding_variance(original_embeddings, ghost_embeddings, ghost_indices)
    B = between_embedding_variance(original_embeddings, ghost_embeddings, ghost_indices)

    V = W + B
    # if not ghost_embeddings is None:
    #     print(np.mean(W), np.mean(B))
    #     print("max, argmax", np.max(W), np.argmax(W))
    #     w = np.argmax(W)
    #     print(original_embeddings[0][w], ghost_embeddings[0][w])
    rank = np.argsort(V)[::-1]
    score = V[rank]
    if ghost_indices is not None:
        rank = ghost_indices[rank]

    return rank, score


def within_embedding_variance(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray = None,
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
    WV: np.ndarray of shape (n_ghost_targets,)
        Mean within-embedding variance for each ghost target.
    """
    if ghost_indices is None:
        ghost_indices = np.arange(original_embeddings.shape[1])
    if ghost_embeddings is None:
        return np.zeros(len(ghost_indices))

    original_embeddings = original_embeddings[:, ghost_indices]
    ghost_embeddings = ghost_embeddings[:, ghost_indices]

    E = original_embeddings[:, :, np.newaxis]
    if ghost_embeddings.shape[2]:
        E = np.concatenate([E, ghost_embeddings], axis=2)
    # shape of E: (n_embeddings, n_samples, n_ghosts+1, n_components)

    Mu = np.mean(E, axis=2)  # shape of M: (n_embeddings, n_samples, n_components)

    W = np.sum(np.square(E - Mu[:, :, np.newaxis]), axis=3)
    W = np.mean(W, axis=(2, 0))

    return W


def between_embedding_variance(
    original_embeddings: np.ndarray,
    ghost_embeddings: np.ndarray = None,
    ghost_indices: np.ndarray = None,
):
    """
    Calculate the variance of the original and ghost points between embeddings.

    Parameters
    ----------
    original_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_components)
    ghost_embeddings: np.ndarray of shape (n_embeddings, n_samples, n_ghosts, n_components)
    ghost_indices: np.ndarray of shape (n_ghost_targets,)

    Returns
    -------
    BV: np.ndarray of shape (n_ghost_targets,)
        Mean between-embedding variance for each ghost target.
    """
    if ghost_indices is None:
        ghost_indices = np.arange(original_embeddings.shape[1])

    original_embeddings = original_embeddings[:, ghost_indices]
    E = original_embeddings[:, :, np.newaxis]

    if ghost_embeddings is not None and ghost_embeddings.shape[2]:
        ghost_embeddings = ghost_embeddings[:, ghost_indices]
        E = np.concatenate([E, ghost_embeddings], axis=2)
    # shape of E: (n_embeddings, n_samples, n_ghosts+1, n_components)

    Mu = np.mean(E, axis=2)  # shape of M: (n_embeddings, n_samples, n_components)

    Eta = np.mean(Mu, axis=0)  # shape of MM: (n_samples, n_components)

    B = np.sum(np.square(Mu - Eta[np.newaxis, :]), axis=2)
    B = np.mean(B, axis=0)

    return B
