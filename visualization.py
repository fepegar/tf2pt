import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

sns.set(context='notebook')

def plot_parameters(model, name, title=None, axis=None, kde=True, bw=None):
    for name_, params in model.named_parameters():
        if name_ == name:
            tensor = params.data
            break
    else:
        raise ValueError(f'{name} not found in model')
    array = tensor.numpy().ravel()
    if axis is None:
        fig, axis = plt.subplots()
    if bw is None:
        sns.distplot(array, ax=axis, kde=kde)
    else:
        sns.kdeplot(array, ax=axis, bw=bw)
    if title is not None:
        ax.set_title(title)

        
def plot_all_parameters(model, labelsize=6, kde=True, bw=None):
    fig, axes = plt.subplots(3, 7, figsize=(11, 5))
    axes = list(reversed(axes.ravel()))
    for name, params in model.named_parameters():
        if len(params.data.shape) < 2:
            continue
        axis = axes.pop()
        plot_parameters(model, name, axis=axis, kde=kde, bw=bw)
        axis.xaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def turn(s):
    return np.fliplr(np.rot90(s))


def rescale_array(array, cutoff=(2, 98)):
    percentiles = tuple(np.percentile(array, cutoff))
    array = rescale_intensity(array, in_range=percentiles)
    return array


def plot_volume(array, enhance=True):
    if array.ndim > 5:
        array = array[0]
    if array.ndim == 5:
        array = array[..., 0, 0]  # 5D to 3D
    if enhance:
        array = rescale_array(array)
    fig, axes = plt.subplots(1, 3, figsize=(9, 5))
    si, sj, sk = array.shape
    slices = [
        array[si//2, ...],
        array[:, sj//2, :],
        array[..., sk//2],
    ]
    cmap = 'gray' if array.ndim == 3 else 'none'
    for i, slice_ in enumerate(slices):
        axis = axes[i]
        axis.imshow(turn(slice_), cmap=cmap)
        axis.grid(False)
    plt.tight_layout()

    
def plot_histogram(array, kde=True):
    sns.distplot(array.ravel(), kde=kde)