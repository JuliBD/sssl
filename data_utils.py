
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102
import torchvision.transforms as transforms
import os

CROP_LOW_SCALE = 0.2
TRAIN_ON_TEST = False
BATCH_SIZE = 1024
N_CPU_WORKERS = 20
DATA_ROOT = "data/"

def get_dataset(dataset_name):
    return dict(
        cifar10 = CIFAR10,
        cifar10_unbalanced = CIFAR10_Unbalanced,
        cifar100 = CIFAR100,
        cifar100_unbalanced = CIFAR100_Unbalanced,
        flowers = Flowers102_Standardized
    )[dataset_name]

def get_train_and_test_set(dataset):
    train = dataset(
        root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    test = dataset(
        root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    return train, test

################################## Augmentations ################################
transforms_ssl = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(CROP_LOW_SCALE, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ]
)
class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item, label = self.dataset[i]

        return i, self.transform(item), self.transform(item), label

def _get_augmented_dataloader(dataset):
    train, test = get_train_and_test_set(dataset)
    return DataLoader(
        AugmentedDataset(
            train + test if TRAIN_ON_TEST else train,
            transforms_ssl,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
    )

def get_augmented_dataloader(dataset_name):
    dataset = get_dataset(dataset_name)
    return _get_augmented_dataloader(dataset)

############################## Datasets ##################################

class CIFAR10_Unbalanced(CIFAR10):
    def __init__(self, root, train, transform, download=False):
        super().__init__(root, train, transform=transform, download=download)
        if train:
            self.throw_away_samples()

    def throw_away_samples(self):
        num_classes = 10
        kept_samples = []
        targets = np.array(self.targets)

        for i, target in enumerate(range(num_classes)):
            target_inds = np.where(targets == target)
            keep_num = int(5000 * np.exp(-i * np.log(1.5))) # Dassot SSL paper from Neurips 2024 SSL workshop
            keep_inds = target_inds[0][:keep_num]
            kept_samples.append(keep_inds)
        kept_samples = np.concatenate(kept_samples)
        self.data = self.data[kept_samples]
        self.targets = targets[kept_samples]


class CIFAR100_Unbalanced(CIFAR100):
    def __init__(self, root, train, transform, download=False):
        super().__init__(root, train, transform=transform, download=download)
        if train:
            self.throw_away_samples()

    def throw_away_samples(self):
        num_classes = 100
        kept_samples = []
        targets = np.array(self.targets)

        for i, target in enumerate(range(num_classes)):
            target_inds = np.where(targets == target)
            keep_num = int(500 * np.exp(-i * np.log(1.03365))) # Dassot SSL paper from Neurips 2024 SSL workshop
            keep_inds = target_inds[0][:keep_num]
            kept_samples.append(keep_inds)
        kept_samples = np.concatenate(kept_samples)
        self.data = self.data[kept_samples]
        self.targets = targets[kept_samples]

from PIL import Image
import torch
class Flowers102_Standardized(Flowers102):
    def __init__(self, root, train, transform, download):
        self.train = train
        self.dataset_size = 128
        # FIXME FIXME FIXME -- Flowers imbalance is in the test set!
        split = 'test' if self.train else 'val'
        super().__init__(root, split, transform=transform, download=download)
        self.targets = torch.tensor(self._labels, dtype=torch.long)

    def __getitem__(self, idx, to_tensor=True):
        img = self._image_files[idx]
        img = Image.open(img).convert("RGB")
        img = img.resize((self.dataset_size, self.dataset_size), Image.BICUBIC)
        if to_tensor:
            img = transforms.ToTensor()(img)
        return img, self.targets[idx]


def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed


cifar10_train, cifar10_test = get_train_and_test_set(CIFAR10)
cifar10_unbalanced_train, cifar10_unbalanced_test = get_train_and_test_set(CIFAR10_Unbalanced)

cifar10_loader_ssl = _get_augmented_dataloader(CIFAR10)
cifar10_unbalanced_loader_ssl = _get_augmented_dataloader(CIFAR10_Unbalanced)
