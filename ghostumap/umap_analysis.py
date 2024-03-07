import json
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


class UmapAnalysis:
    def __init__(
        self,
        data: np.ndarray,
        label: np.ndarray | list = None,
        title: str = "mnist",
        n_neighbor: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
    ):
        self.data = data
        self.label = label
        self.title = title

        # umap hyperparameters
        self.n_neighbor = n_neighbor
        self.min_dist = min_dist
        self.n_epochs = n_epochs

        self.var_ranks = None
        self.var_scores = None

    def generate_samples(self, n_samples: int = 10):
        self.n_samples = n_samples

        self.reducer: UMAP = UMAP(
            n_neighbors=self.n_neighbor,
            min_dist=self.min_dist,
            n_epochs=[i for i in range(self.n_epochs)],
        )
        projection = self.reducer.fit_transform(self.data)

        (self.knn_indices, self.knn_distances, self.knn_search_index) = (
            self.reducer._knn_indices,
            self.reducer._knn_dists,
            self.reducer._knn_search_index,
        )
        self.reducer: UMAP = UMAP(
            n_neighbors=self.n_neighbor,
            min_dist=self.min_dist,
            n_epochs=self.n_epochs,
            precomputed_knn=(
                self.knn_indices,
                self.knn_distances,
                self.knn_search_index,
            ),
        )
        self.projection_samples = np.array(
            [projection]
            + [self.reducer.fit_transform(self.data) for _ in range(self.n_samples - 1)]
        )

    def save_results(self, result_path: str = "./result/umap_analysis/mnist"):
        """
        save results
        config : umap hyperparameters
        projection_samples : projection samples
        data label: data label
        """
        if os.path.exists(result_path) == False:
            os.makedirs(result_path)

        np.save(
            os.path.join(result_path, "projection_samples.npy"), self.projection_samples
        )
        np.save(os.path.join(result_path, "knn_indices.npy"), self.knn_indices)
        np.save(os.path.join(result_path, "knn_distances.npy"), self.knn_distances)

        config = {
            "n_samples": self.n_samples,
            "n_neighbor": self.n_neighbor,
            "min_dist": self.min_dist,
            "n_epochs": self.n_epochs,
        }
        with open(os.path.join(result_path, "config.json"), "w") as f:
            json.dump(config, f)

        var_json = {
            "variance_ranks": self.var_ranks.tolist(),
            "variance_scores": self.var_scores.tolist(),
        }
        with open(os.path.join(result_path, "sample_variances.json"), "w") as f:
            json.dump(var_json, f)

    def load_results(self, result_path: str = "./result/umap_analysis/mnist"):
        """
        load results
        """
        if os.path.exists(result_path) == False:
            raise FileNotFoundError(f"{result_path} does not exist")

        self.projection_samples = np.load(
            os.path.join(result_path, "projection_samples.npy")
        )
        self.knn_indices = np.load(os.path.join(result_path, "knn_indices.npy"))
        self.knn_distances = np.load(os.path.join(result_path, "knn_distances.npy"))
        with open(os.path.join(result_path, "config.json"), "r") as f:
            config = json.load(f)
            self.n_samples = config["n_samples"]
            self.n_neighbor = config["n_neighbor"]
            self.min_dist = config["min_dist"]
            self.n_epochs = config["n_epochs"]

        with open(os.path.join(result_path, "sample_variances.json"), "r") as f:
            var_json = json.load(f)
            self.var_ranks = var_json["variance_ranks"]
            self.var_scores = var_json["variance_scores"]

    def calculate_variance_score(self):
        """
        calculate anomaly score
        by using variance of distance between nearest neighbors among samples
        """

        mean_of_knn_dist_var = np.array(
            [
                np.mean(
                    np.array(
                        [
                            np.var(
                                np.linalg.norm(
                                    self.projection_samples[:, knn_idx]
                                    - self.projection_samples[:, idx],
                                    axis=1,
                                )
                            )
                            for knn_idx in knn
                        ]
                    )
                )
                for idx, knn in enumerate(self.knn_indices)
            ]
        )

        self.var_ranks = np.argsort(mean_of_knn_dist_var)[::-1]
        self.var_scores = mean_of_knn_dist_var[self.var_ranks]

        return self.var_ranks, self.var_scores
