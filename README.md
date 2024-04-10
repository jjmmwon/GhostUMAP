# GhostUMAP

### Installation

```Bash
git clone https://github.com/jjmmwon/ghostumap.git
cd ghostumap
hatch shell
```

### How to use GhostUMAP
```Python
from ghostumap import GhostUMAP
from sklearn.datasets import load_digits

digits = load_digits()

results = GhostUMAP().fit_transform(digits.data, n_embeddings=1, n_ghosts=4)
```

There are two parameters that are different from umap for the fit_transform method in the GhostUMAP class as follows:
> - `n_embeddings`: This determins the number of embeddings.
> - `n_ghosts`: This determins the number of ghosts.
