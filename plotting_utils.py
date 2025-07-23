import matplotlib.pyplot as plt
from data_utils import cifar10_train, cifar10_unbalanced_train
import numpy as np

def plot_variant_norm_hists(dict_of_embed_dict):
    """Plots a norm historgram for each variant in dict_of_embed_dict
    """
    n = len(list(dict_of_embed_dict.keys()))
    fig = plt.figure()
    gs = fig.add_gridspec(n, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for i, (variant_name, embed_dict) in enumerate(dict_of_embed_dict.items()):
        embeds = embed_dict["projector"]
        norms = np.linalg.norm(embeds,axis=-1)
        ax = axs[i]
        counts, bins = np.histogram(norms, "auto")
        ax.stairs(counts, bins, fill=True, label=variant_name)
        ax.legend(loc='upper right')

    plt.show()

def plot_class_norm_hist(dict_of_embed_dict, dataset_for_class_names = cifar10_train):
    """Plots a norm historgram for each class in each variant in dict_of_embed_dict
    """
    label_name = {idx: cls for cls, idx in dataset_for_class_names.class_to_idx.items()}
    n_label = len(list(label_name.keys()))

    n = len(list(dict_of_embed_dict.keys()))
    fig = plt.figure()
    gs = fig.add_gridspec(n, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    if n == 1:
        axs = [axs]

    for i, (variant_name, embed_dict) in enumerate(dict_of_embed_dict.items()):
        embeds = embed_dict["projector"]
        labels = embed_dict["labels"]
        norms = np.linalg.norm(embeds,axis=-1)
        ax = axs[i]
        ax.set_title(variant_name)
        for idx, label in label_name.items():
            class_norms = norms[labels==idx]
            counts, bins = np.histogram(class_norms, "auto")
            ax.stairs(counts, bins, fill=True, label=label, alpha=0.3)
            ax.legend(loc='upper right')

    plt.show()


def plot_extrem_norm_images_per_class(embed_dict, high_norm=False, n_image_per_class=10, dataset_for_class_names=cifar10_train):
    # Label names
    label_name = {idx: cls for cls, idx in dataset_for_class_names.class_to_idx.items()}

    # Initialize storage for 10 images per class
    images_per_label = {i: None for i in range(len(list(label_name.keys())))}
    all_embed_norms = np.linalg.norm(embed_dict["projector"], axis=1)

    # Collect 10 images per class
    for idx in label_name.keys():
        label_idx = embed_dict["labels"]==idx
        all_images_with_label = embed_dict["images"][label_idx]
        embed_norms = all_embed_norms[label_idx]
        sort_idx_ascending = embed_norms.argsort()
        sort_idx_descending = sort_idx_ascending[::-1]
        sort_images = all_images_with_label[sort_idx_descending if high_norm else sort_idx_ascending]
        images_per_label[idx] = sort_images[:n_image_per_class]
            

    # Set up a 10x10 grid
    fig, axes = plt.subplots(10, n_image_per_class, figsize=(n_image_per_class, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    # Plot the images by class
    for class_idx in range(10):
        for i in range(n_image_per_class):
            image = images_per_label[class_idx][i]
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)

            ax = axes[class_idx, i]
            ax.imshow(image)

            # Add class name on the leftmost image in each row
            if i == 0:
                ax.annotate(label_name[class_idx], xy=(0, 0.5), xycoords='axes fraction',
                            fontsize=8, ha='right', va='center', rotation=0,
                            xytext=(-5, 0), textcoords='offset points')
            ax.axis('off')
    
    fig.suptitle(f'Cifar10, for each class the images with the *{"highest" if high_norm else "lowest"}* embedding norm are shown', fontsize=16)
    plt.show()


def plot_images_per_class_with_norm_close_to_0_intersect_of_extra_dim(
        embed_dict,
        high_dist=False,
        n_image_per_class=10,
        dataset_for_class_names=cifar10_train,
        zero_intersect=1,
        
        ):
    # Label names
    label_name = {idx: cls for cls, idx in dataset_for_class_names.class_to_idx.items()}

    # Initialize storage for 10 images per class
    images_per_label = {i: None for i in range(len(list(label_name.keys())))}
    all_embed_norms = np.linalg.norm(embed_dict["projector"], axis=1)

    # Collect 10 images per class
    for idx in label_name.keys():
        label_idx = embed_dict["labels"]==idx
        all_images_with_label = embed_dict["images"][label_idx]
        embed_norms = all_embed_norms[label_idx]
        dist_to_zero_intersect = np.abs(embed_norms-zero_intersect)
        sort_idx_ascending = dist_to_zero_intersect.argsort()
        sort_idx_descending = sort_idx_ascending[::-1]
        sort_images = all_images_with_label[sort_idx_descending if high_dist else sort_idx_ascending]
        images_per_label[idx] = sort_images[:n_image_per_class]
            
    # Set up a 10x10 grid
    fig, axes = plt.subplots(10, n_image_per_class, figsize=(n_image_per_class, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    # Plot the images by class
    for class_idx in range(10):
        for i in range(n_image_per_class):
            image = images_per_label[class_idx][i]
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)

            ax = axes[class_idx, i]
            ax.imshow(image)

            # Add class name on the leftmost image in each row
            if i == 0:
                ax.annotate(label_name[class_idx], xy=(0, 0.5), xycoords='axes fraction',
                            fontsize=8, ha='right', va='center', rotation=0,
                            xytext=(-5, 0), textcoords='offset points')
                
            ax.axis('off')
    
    fig.suptitle(f'Cifar10, for each class the images with the *{"highest" if high_dist else "lowest"}* dist to the zero intersect (={zero_intersect}) are shown', fontsize=16)
    plt.show()

def mean_cos_sim_between_embeddings_with_extreme_norm(embed_dict, k=20, print_class_mean_cos_sim = False, use_embeds_mapped_onto_sphere=True):
    embeds = embed_dict["projector"]
    norms = np.linalg.norm(embeds, axis=1)
    if use_embeds_mapped_onto_sphere:
        normalized_embds = embed_dict["normalized"].cpu().numpy()
    else:
        normalized_embds = embeds/norms.reshape(-1, 1)
    labels = embed_dict["labels"]

    label_name = {idx: cls for cls, idx in cifar10_train.class_to_idx.items()}

    def mean_cos_sim(embeds):
        cosine_sim_matrix = embeds @ embeds.T
        batch_dim = embeds.shape[0]
        mask_out_diagonal = ~np.eye(batch_dim, dtype=bool)
        mean_cosine_similarity = cosine_sim_matrix[mask_out_diagonal].mean()
        return mean_cosine_similarity


    high_norm_cos_sim_sum = 0
    low_norm_cos_sim_sum = 0

    for idx, cls in label_name.items():
        cls_samples = labels == idx
        cls_norms = norms[cls_samples]
        cls_normalized_embds = normalized_embds[cls_samples]
        sort_idx = cls_norms.argsort()

        high_norm_k = cls_normalized_embds[sort_idx[::-1][:k]]
        high_norm_mean_cosine_similarity = mean_cos_sim(high_norm_k)
        high_norm_cos_sim_sum += high_norm_mean_cosine_similarity
        
        low_norm_k = cls_normalized_embds[sort_idx[:k]]
        low_norm_mean_cosine_similarity = mean_cos_sim(low_norm_k)
        low_norm_cos_sim_sum += low_norm_mean_cosine_similarity
        
        if print_class_mean_cos_sim:
            print(f"mean cosine similarity between all {k} datapoints in *{cls}* with")
            print(f"highest norm:", high_norm_mean_cosine_similarity)
            print("lowest norm:", low_norm_mean_cosine_similarity)
            print()

    n_classes = len(list(label_name.keys()))
    print("high norm mean over classes:", high_norm_cos_sim_sum/n_classes)
    print("low norm mean over classes:", low_norm_cos_sim_sum/n_classes)
    print()


    sort_idx = norms.argsort()
    k = k * n_classes
    print(f"mean cosine similarity between all {k} datapoints with")

    high_norm_k = normalized_embds[sort_idx[::-1][:k]]

    low_norm_k = normalized_embds[sort_idx[:k]]
    print(f"highest norm:", mean_cos_sim(high_norm_k))
    print("lowest norm:", mean_cos_sim(low_norm_k))
    print()


import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection, PathCollection


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))

    return ax.add_collection(lc)

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

def mean_over_extreme_points_dist_norm_epoch(
        ax,
        embed_hist,
        starting_epoch=0,
        moving_average_size=1,
        extreme_at_end_in : Literal['norm', 'distance'] = "norm",
        n_extrem_points = 100,
        line_or_scatter : Literal['line', 'scatter'] = "line",
        only_mean = False
        ):
    
    norm_history = embed_hist.item()["norm_history"][:-1]
    dist_history = embed_hist.item()["distance_history"]
    final_epoch = len(dist_history)

    # sort by norm or dist in last epoch [-1]
    if extreme_at_end_in == "norm":
        sort_by = norm_history[-1].argsort()
    elif extreme_at_end_in == "distance":
        sort_by = dist_history[-1].argsort()
        
    def moving_average(x, window_size):
        return np.convolve(x, np.ones(window_size), 'valid') / window_size
    
    low_x = moving_average(norm_history[starting_epoch:,sort_by[:n_extrem_points]].mean(axis=1), moving_average_size)
    low_y = moving_average(dist_history[starting_epoch:,sort_by[:n_extrem_points]].mean(axis=1), moving_average_size)

    high_x = moving_average(norm_history[starting_epoch:,sort_by[-n_extrem_points:]].mean(axis=1), moving_average_size)
    high_y = moving_average(dist_history[starting_epoch:,sort_by[-n_extrem_points:]].mean(axis=1), moving_average_size)
    
    mean_x = moving_average(norm_history[starting_epoch:].mean(axis=1), moving_average_size)
    mean_y = moving_average(dist_history[starting_epoch:].mean(axis=1), moving_average_size)

    color = np.linspace(starting_epoch, final_epoch, len(low_x))
    cmap = "plasma"
    alpha = 0.7
    s = 10
    edgecolor = "none"
    if line_or_scatter == "scatter":
        if not only_mean:
            low = ax.scatter(low_x, low_y, c=color, s=s, cmap=cmap, alpha=alpha, edgecolor=edgecolor)
            high = ax.scatter(high_x, high_y, c=color, s=s, cmap=cmap, alpha=alpha, edgecolor=edgecolor)
        mean = ax.scatter(mean_x, mean_y, c=color, s=s, cmap=cmap, alpha=alpha, edgecolor=edgecolor)
    elif line_or_scatter == "line":
        if not only_mean:
            low = colored_line(low_x, low_y, c=color, ax=ax, cmap=cmap, alpha=alpha)
            high = colored_line(high_x, high_y, c=color, ax=ax, cmap=cmap, alpha=alpha)
        mean = colored_line(mean_x, mean_y, c=color, ax=ax, cmap=cmap, alpha=alpha)


    all_max_x = 0
    x_axis = [mean_x] if only_mean else [low_x, high_x, mean_x]
    for xs in x_axis:
        xs_max = xs.max()
        if xs_max > all_max_x:
            all_max_x = xs_max

    all_max_y = 0
    y_axis = [mean_y] if only_mean else [low_y, high_y, mean_y]
    for ys in y_axis:
        ys_max = ys.max()
        if ys_max > all_max_y:
            all_max_y = ys_max

    ax.set_xlim(0, all_max_x)
    ax.set_ylim(0, all_max_y)
    return mean

def plot_mean_over_extreme_points_dist_norm_epoch(
        embed_hist,
        starting_epoch=0,
        moving_average_size=1,
        extreme_at_end_in = "norm",
        n_extrem_points = 100
        ):
    
    fig, ax = plt.subplots()
    lines = mean_over_extreme_points_dist_norm_epoch(
        ax,
        embed_hist,
        starting_epoch,
        moving_average_size,
        extreme_at_end_in,
        n_extrem_points
    )
    ax.set_xlabel("norm: mean across datapoints")
    ax.set_ylabel("angular distance: mean across datapoints")
    fig.colorbar(lines, label="epoch")
    plt.show()


def plot_paths_mean_over_extreme_points_dist_norm_epoch(
        paths: dict,
        starting_epoch = 0,
        moving_average_size = 1,
        n_extreme_points = 200,
        extreme_at_end_in = "norm",
        line_or_scatter : Literal['line', 'scatter'] = "line",
        share_axis = False,
        only_mean = False
    ):

    n = len(list(paths.keys()))
    fig = plt.figure(figsize=(n*5, 6))
    gs = fig.add_gridspec(1,n, hspace=0)
    axs = gs.subplots(sharex=share_axis, sharey=share_axis)

    all_max_x = 0
    all_max_y = 0

    for ax, (variant_name, variant_path) in zip(axs, paths.items()):
        with open(variant_path, "rb") as f:
            embed_hist = np.load(f, allow_pickle=True)
        colored_plot = mean_over_extreme_points_dist_norm_epoch(ax, embed_hist, starting_epoch, moving_average_size, extreme_at_end_in, n_extreme_points, line_or_scatter, only_mean)
        ax.set_title(variant_name)

        all_max_x = max(ax.get_xlim()[1], all_max_x)
        all_max_y = max(ax.get_ylim()[1], all_max_y)

    if share_axis:
        ax.set_xlim(0, all_max_x)
        ax.set_ylim(0, all_max_y)

    axs[0].set_xlabel("embedding norm: mean across datapoints")
    for ax in axs[1:]:
        ax.set_xlabel("embedding norm")
    axs[0].set_ylabel("angular distance: mean across datapoints")

    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # for ax in axs.flat[1:]:
    #     ax.spines['left'].set_visible(False)

    # fig.suptitle(f"Comparison of training dynamic\n extrem points with regards to {extreme_at_end_in}")
    fig.colorbar(colored_plot, label="epoch")
    plt.show()

from matplotlib.animation import FuncAnimation, PillowWriter

def gif_distance_norm(embed_history_path):
    # e.g. embed_history_path = "logs/Point_tracking-1000/l2/version_5/embed_history.npy"
    with open(embed_history_path, "rb") as f:
        embed_hist = np.load(f, allow_pickle=True)

    moving_average_size = 20
    plot_every_n_epochs = 10
    start_at_epoch = 50

    def moving_average(x, window_size=moving_average_size):
        return np.convolve(x, np.ones(window_size), 'valid') / window_size

    norm_history = np.apply_along_axis(moving_average, 1, embed_hist.item()["norm_history"])
    dist_history = np.apply_along_axis(moving_average, 1, embed_hist.item()["distance_history"])

    n = norm_history.shape[1]

    n_epochs = dist_history.shape[0]
    epoch_dist = {}
    epoch_norm = {}


    highlight_dicts = dict(
        high_norm_at_end = dict(idx=norm_history[-1].argsort()[-10:], color="red"),
        low_norm_at_end = dict(idx=norm_history[-1].argsort()[:10], color="green"),
        median_norm_at_end = dict(idx=norm_history[-1].argsort()[int(n/2)-5:int(n/2)+5], color="black")
    )
    for dic in highlight_dicts.values():
        dic["dist"] = {}
        dic["norm"] = {}

    epochs_displayed = list(range(start_at_epoch, n_epochs, plot_every_n_epochs))
    if epochs_displayed[-1] < n_epochs:
        epochs_displayed.append(n_epochs-1)

    max_norm = 0
    min_norm = 0
    max_dist = 0
    min_dist = 0

    epoch_up_to = {}
    for i in epochs_displayed:
        dist_mean = dist_history[i].mean()
        norm_mean = norm_history[i].mean()

        epoch_dist[i] = dist_history[i] - dist_mean
        epoch_norm[i] = norm_history[i] - norm_mean


        for dic in highlight_dicts.values():
            highlighted_idx = dic["idx"]
            dic["dist"][i] = dist_history[i, highlighted_idx] - dist_mean
            dic["norm"][i] = norm_history[i, highlighted_idx] - norm_mean

        max_norm = max(epoch_norm[i].max(), max_norm)
        min_norm = min(epoch_norm[i].min(), min_norm)
        max_dist = max(epoch_dist[i].max(), max_dist)
        min_dist = min(epoch_dist[i].min(), min_dist)

    fig = plt.figure()
    gs = fig.add_gridspec(1, hspace=0)
    ax = gs.subplots()
    initial_plot_n = epochs_displayed[0]

    scatter_all = ax.scatter(epoch_norm[initial_plot_n], epoch_dist[initial_plot_n], alpha=0.01, edgecolors="none")

    for name, dic in highlight_dicts.items():
        dic["scatter"] = ax.scatter(
            dic["norm"][initial_plot_n],
            dic["dist"][initial_plot_n],
            alpha=0.5,
            edgecolors="none",
            color=dic["color"],
            label=name
            )

    fig.suptitle(f"Epoch: {start_at_epoch}\n Moving avererage size: {moving_average_size}")
    ax.set_xlabel(f"embedding norm - mean")
    ax.set_ylabel(f"angular distance - mean")

    ax.set_ylim((min_dist,max_dist))
    ax.set_xlim((min_norm,max_norm))
    ax.legend()


    def animate(i):
        epoch=epochs_displayed[i]
        scatter_all.set_offsets(np.c_[epoch_norm[epoch], epoch_dist[epoch]])

        for dic in highlight_dicts.values():
            highlighted_norm = dic["norm"]
            highlighted_dist = dic["dist"]
            dic["scatter"].set_offsets(np.c_[highlighted_norm[epoch], highlighted_dist[epoch]])

        fig.suptitle(f"Epoch: {epoch}\n Moving avererage size: {moving_average_size}")
        return [scatter_all] + [dic["scatter"] for dic in highlight_dicts.values()]

    ani = FuncAnimation(fig, animate, interval=1, blit=True, repeat=True, frames=len(epochs_displayed)) #len(epochs_displayed)
    ani.save("distance-norm.gif", dpi=300, writer=PillowWriter(fps=10))


def plot_dim_hist(embeds, display_mode="show_fist_two"):
    """Plots a histogram for each embedding dimension
    """
    dims_to_show = dict(
        show_fist_two = [0,1],
        show_all = list(range(embeds.shape[1])),
        show_sample = [i**2 for i in range(embeds.shape[1]) if i**2 < embeds.shape[1]]
    )[display_mode]
    
    n_selected = len(dims_to_show)

    fig = plt.figure()
    gs = fig.add_gridspec(n_selected, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    for i, dim in enumerate(dims_to_show):
        counts, bins = np.histogram(embeds[:,dim], "auto")
        ax = axs[i]
        ax.stairs(counts, bins, fill=True, label=str(dim))
        ax.legend(loc='upper right')
    plt.show()

######################################## dealing with checkpoints and getting embeddings ##################
import torch
import lightning_simclr as ls
import re, os
from torch.utils.data import Dataset, DataLoader

def get_ckpts(ckpt_folder):

    epoch_ckpt = {}
    for path in os.listdir(ckpt_folder):
        matches = re.search(r"epoch=(\d+)", path)
        epoch_ckpt[int(matches.group(1))] = os.path.join(ckpt_folder, matches.group(0)+".ckpt")
    
    return epoch_ckpt

def _get_embeddings(model, dataset, device):
    I = []
    X = []
    y = []
    Z = []

    for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1)):
        images, labels = batch

        h, z = model(images.to(device))

        I.append(images.cpu().numpy())
        X.append(h.cpu().numpy())
        Z.append(z.cpu().numpy())
        y.append(labels)

    I = np.vstack(I)
    X = np.vstack(X)
    Z = np.vstack(Z)
    y = np.hstack(y)

    return I, X, y, Z

def get_embeddings(mod, dataset=None):
    mod.model.eval()
    dataset = mod.train_dataset if not dataset else dataset
    with torch.no_grad():
        I, X, y, Z = _get_embeddings(mod.model, dataset, mod.device)
    mod.model.train()
    return dict(
        images=I,
        backbone=X,
        projector=Z,
        normalized=mod.norm_function(torch.tensor(Z)).cpu().numpy(),
        labels=y
    )

def get_embeds_for_ckpt(ckpt_path, dataset=None):
    mod = ls.SimCLR.load_from_checkpoint(ckpt_path)
    return get_embeddings(mod, dataset)


def get_embeds_for_ckpts(ckpt_folder, dataset=None):
    ckpts = get_ckpts(ckpt_folder)
    ckpt_embeds = {}
    for epoch, path in ckpts.items():
        ckpt_embeds[epoch] = get_embeds_for_ckpt(path, dataset)
        
    return ckpt_embeds