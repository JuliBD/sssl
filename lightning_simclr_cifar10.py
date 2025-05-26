import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import ConstantLR, ChainedScheduler
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torchvision.transforms as transforms

import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from norm_functions import *

###################### PARAMS ##############################

BATCH_SIZE = 1024
N_EPOCHS = 1000
N_CPU_WORKERS = 20
BASE_LR = 0.06
WEIGHT_DECAY = 5e-4
PROJECTOR_HIDDEN_SIZE = 1024
CROP_LOW_SCALE = 0.2
NORM_FUNCTION_STR = "l2"
CLAMP = False
NESTEROV = False
TRAIN_ON_TEST = True


NORM_FUNCTION_DICT = {
                        'l2': F.normalize, # if no norm_function_str is provided this will be used (default SimCLR)
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        'stereo': stereo_normalize,
                        "torus": torus_norm,
                    }


###################### DATA LOADER #########################

cifar10_train = CIFAR10(
    root=".", train=True, download=True, transform=transforms.ToTensor()
)
cifar10_test = CIFAR10(
    root=".", train=False, transform=transforms.ToTensor()
)

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

        return self.transform(item), self.transform(item), label


cifar10_loader_ssl = DataLoader(
    AugmentedDataset(
        cifar10_train if not TRAIN_ON_TEST else cifar10_train + cifar10_test,
        transforms_ssl,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_CPU_WORKERS,
)


###################### NETWORK ARCHITECTURE #########################


class ResNet18withProjector(nn.Module):
    def __init__(self,
                 
                 ):
        super().__init__()

        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, PROJECTOR_HIDDEN_SIZE), 
            nn.ReLU(), 
            nn.Linear(PROJECTOR_HIDDEN_SIZE, 128),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def infoNCE(
        features, 
        temperature=0.5, 
        norm_function=F.normalize
        ):
    batch_size = features.size(0) // 2

    x = norm_function(features)
    cos_xx = x @ x.T / temperature

    cos_xx.fill_diagonal_(float("-inf"))
        
    targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)


class CosineAnnealingWarmup(CosineAnnealingLR):
    def __init__(
            self,
            optimizer,
            T_max,
            eta_min = 0,
            last_epoch = -1,
            verbose="deprecated",
            warmup_epochs=10,
            warmup_lr=0
            ):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.cur_epoch = 0
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)
    
    def get_lr(self):
        self.cur_epoch += 1
        cur_epoch = self.cur_epoch
        warmup_epochs = self.warmup_epochs
        warmup_lr = self.warmup_lr

        if cur_epoch < warmup_epochs:
            # linear reascaling of lr between warmup and the cosine annealing lr value
            out = [((warmup_epochs-cur_epoch)/warmup_epochs) * warmup_lr * group
                    + cur_epoch/warmup_epochs * group
                    for group in super().get_lr()]
            self.cur_epoch = self.cur_epoch + 1
            return out
        else:
            return super().get_lr()

class SimCLR(pl.LightningModule):

    def __init__(
            self,
            base_lr = BASE_LR,
            batch_size = BATCH_SIZE,
            weight_decay = WEIGHT_DECAY,
            nestrov = NESTEROV,
            n_epochs = N_EPOCHS,
            norm_function_str = NORM_FUNCTION_STR
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet18withProjector()
        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]
        self.loss = lambda features: infoNCE(features, norm_function=self.norm_function)

        self.base_lr = base_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.nestrov = nestrov
        self.n_epochs = n_epochs
        self.norm_function_str = norm_function_str
    
    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.base_lr * self.batch_size / 256,
            momentum=0.9,
            weight_decay=self.weight_decay,
            nesterov=self.weight_decay,
        )
        
        scheduler = CosineAnnealingWarmup(optimizer, T_max=self.n_epochs)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):
        view1, view2, _ = batch
        view1 = view1
        view2 = view2

        _, z1 = self.model(view1)
        _, z2 = self.model(view2)

        loss = self.loss(torch.cat((z1, z2)))

        if len(batch) == self.batch_size: # only record batches with full size
            self.log("loss", loss, prog_bar=True)
        return loss


###################### FINAL EVALUATION #########################


def dataset_to_X_y(dataset, model):
    X = []
    y = []
    Z = []

    for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1024)):
        images, labels = batch

        h, z = model(images.to(model.device))

        X.append(h.cpu().numpy())
        Z.append(z.cpu().numpy())
        y.append(labels)

    X = np.vstack(X)
    Z = np.vstack(Z)
    y = np.hstack(y)

    return X, y, Z

def eval(pl_module, log_dir):
    pl_module.eval()
    eval_results = {}
    print("Computing features for evaluation")
    with torch.no_grad():
        X_train, y_train, Z_train = dataset_to_X_y(cifar10_train, pl_module)
        X_test, y_test, Z_test = dataset_to_X_y(cifar10_test, pl_module)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    eval_results["kNN accuracy (Euclidean)"] = score
    print(f"kNN accuracy (Euclidean): {score}", flush=True)

    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    eval_results["kNN accuracy (cosine)"] = score
    print(f"kNN accuracy (cosine): {score:.4f}", flush=True)
    
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    knn.fit(Z_train, y_train)
    score = knn.score(Z_test, y_test)
    eval_results["kNN accuracy (cosine), Z"] = score
    print(f"kNN accuracy (cosine), Z: {score:.4f}", flush=True)
    
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    norm_function = pl_module.norm_function
    Z_train_norm = norm_function(torch.tensor(Z_train))
    Z_test_norm = norm_function(torch.tensor(Z_test))
    knn.fit(Z_train_norm, y_train)
    score = knn.score(Z_test_norm, y_test)
    eval_results["kNN accuracy (cosine), Z after normalization"] = score
    print(f"kNN accuracy (cosine), Z after normalization: {score:.4f}", flush=True)

    lin = LogisticRegression(penalty=None, solver="saga")
    lin.fit(X_train, y_train)
    score = lin.score(X_test, y_test)
    eval_results["Linear accuracy (saga/no-penalty)"] = score
    print(f"Linear accuracy (saga/no-penalty): {score}", flush=True)

    # lin = LogisticRegression(solver="saga")
    # lin.fit(X_train, y_train)
    # score = lin.score(X_test, y_test)
    # eval_results["Linear accuracy (saga)"] = score
    # print(f"Linear accuracy (saga): {score}", flush=True)

    # lin = LogisticRegression(penalty=None)
    # lin.fit(X_train, y_train)
    # score = lin.score(X_test, y_test)
    # eval_results["Linear accuracy (no penalty)"] = score
    # print(f"Linear accuracy (no penalty): {score}", flush=True)

    # lin = LogisticRegression()
    # lin.fit(X_train, y_train)
    # score = lin.score(X_test, y_test)
    # eval_results["Linear accuracy (all defaults)"] = score
    # print(f"Linear accuracy (all defaults): {score}", flush=True)


    np.save(log_dir+"/accuracies", eval_results)


###################### ACTUAL COMPUTING #########################


from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
#torch.set_float32_matmul_precision('high') # 'medium' even high cuased numerical issues
logdir = "logs"

def train_variants(variants):

    for variant in variants:
        pl_module = SimCLR(**variant)
        logger = TensorBoardLogger(logdir, name=f"SimCLR-{pl_module.norm_function_str}")
        trainer = pl.Trainer(max_epochs=N_EPOCHS, logger=logger)
        trainer.fit(pl_module, cifar10_loader_ssl)
        eval(pl_module, logger.log_dir)

variants = [
    dict(norm_function_str = "exp_map"),
    dict(norm_function_str = "l2"),
]

#train_variants(variants)