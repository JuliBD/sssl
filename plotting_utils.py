import matplotlib.pyplot as plt
from data_utils import cifar10_train, cifar10_unbalanced_train
import numpy as np

def plot_variant_norm_hists(dict_of_embed_dict):
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