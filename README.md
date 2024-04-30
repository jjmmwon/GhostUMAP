<p align="center">
  <h2 align="center">GhostUMAP</h2>
	<h3 align="center">Measuring Pointwise Instability in Dimensionality Reduction</h3>
</p>

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

O, G, ghost_indices = GhostUMAP().fit_transform(digits.data, n_ghosts=4, schedule=[50, 100, 150])

```


## API
### Function 'fit_transform'
```Python
def fit_transform(X, n_ghosts, schedule):
```
Fit X into embedded space with ghosts which are clones of each embedded point. They may be located in a different position from their original point due to the stochasticity of UMAP. The variance of these points represent the instability of each point.

#### Parameters
There are two parameters for the ```fit_transform``` method in the GhostUMAP as follows:
> - `n_ghosts`: This parameter sets the number of ghosts. Larger values enable more precise measurements of pointwise instability but come with higher computational costs. The default value is 8, though we generally recommend a range of 5 to 20.
> - `schedule`: This parameter defines a schedule for the successive halving of ghosts. During the optimization process, the instability of each ghost is assessed, and stable ghosts are discarded at the specified epochs in this schedule. The timing of the initial successive halving (SH) presents a tradeoff between time and accuracy: SHing too early may lead to the premature discarding of unstable ghosts, compromising accuracy, whereas SHing too late can prolong execution time.

#### Returns

```O: array, shape (n_samples, n_components)```
Embedding of the original data points, identical to the output of UMAP. It represents the transformed coordinates in the low-dimensional space.

```G: array, shape (n_samples, n_ghosts, n_components)```
Embedding of ghost points which are clones of the original points. These ghost points are used to evaluate the instability of each data instance.

```ghost_inices: array, shape (n_remaining_ghosts,)``` 
This array lists the indices of the remaining ghost points after the successive halving process.


### Function 'measure_instability'
```Python
def measure_instability():
```
This function assesses and ranks the instability of ghost points based on their variance in the low-dimensional space. It should be invoked after the fit_transform function.

#### Returns
```unstable_ghosts: array,shape (n_remaining_ghosts,)```
This array contains indices of the ghost points sorted by their instability. Indices at the beginning of the array represent the most unstable ghosts.

```instabilities: array, shape (n_remaining_ghosts,)```
This array holds the instability values corresponding to the indices in the unstable_ghosts array, providing a direct measure of each ghost's variance.

### Function 'plot'
```Python
def plot(ghost_idx):
```
This function plots the original embedding with the ghost points as a scatterplot.

#### Parameter
```ghost_idx : int```
The index of the ghost point to plot.

#### Returns
```fig, ax```
It returns matplotlib objects.




