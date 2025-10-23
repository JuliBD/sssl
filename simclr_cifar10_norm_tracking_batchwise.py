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
from sklearn.linear_model import LogisticRegression
import pandas as pd
import utils.losses as losses

###################### PARAMS ##############################

BATCH_SIZE = 1024
N_ANNEAL_EPOCHS = 1000
N_EPOCHS = 20
N_CPU_WORKERS = 20
BASE_LR = 0.06
WEIGHT_DECAY = 5e-4
PROJECTOR_HIDDEN_SIZE = 1024
CROP_LOW_SCALE = 0.2
CLAMP = False
NESTEROV = False
MOMENTUM = 0#0.9

ATTRACTION_FACTOR = 1
REPULSION_FACTOR = 1

PRINT_EVERY_EPOCHS = 1
TRAIN_ON_TEST = False

embed_histories_path = "batch_metrics/sim_clr_histories_no_momentum.pkl"


hyper_param_dict =dict(
    BATCH_SIZE = BATCH_SIZE,
    N_ANNEAL_EPOCHS = N_ANNEAL_EPOCHS,
    N_EPOCHS = N_EPOCHS,
    N_CPU_WORKERS = N_CPU_WORKERS,
    BASE_LR = BASE_LR,
    WEIGHT_DECAY = WEIGHT_DECAY,
    PROJECTOR_HIDDEN_SIZE = PROJECTOR_HIDDEN_SIZE,
    CROP_LOW_SCALE = CROP_LOW_SCALE,
    CLAMP = CLAMP,
    NESTEROV = NESTEROV,
    MOMENTUM = MOMENTUM,
    PRINT_EVERY_EPOCHS = PRINT_EVERY_EPOCHS,
    TRAIN_ON_TEST = TRAIN_ON_TEST,
    ATTRACTION_FACTOR = ATTRACTION_FACTOR,
    REPULSION_FACTOR = REPULSION_FACTOR,
)
###################### DATA LOADER #########################

cifar10_train = CIFAR10(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)
cifar10_test = CIFAR10(
    root="data", train=False, transform=transforms.ToTensor()
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

        return i, item, label, self.transform(item), self.transform(item)

cifar10_loader_ssl = DataLoader(
    AugmentedDataset(
        cifar10_train if not TRAIN_ON_TEST else cifar10_train + cifar10_test,
        transforms_ssl,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_CPU_WORKERS,
)


class IndexdDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item, label = self.dataset[index]
        return index, item, label


cifar10_indexed_train = IndexdDataset(cifar10_train)
cifar10_indexed_test = IndexdDataset(cifar10_test)



###################### NETWORK ARCHITECTURE #########################


class ResNet18withProjector(nn.Module):
    def __init__(self):
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
        self.embed = z # save for monitoring
        return h, z


def infoNCE(features, temperature=0.5):
    batch_size = features.size(0) // 2

    x = F.normalize(features)
    cos_xx = x @ x.T / temperature

    self_mask = torch.eye(cos_xx.size(0), dtype=bool, device=cos_xx.device)
    cos_xx.masked_fill_(self_mask, float("-inf"))

    targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)

model = ResNet18withProjector()

optimizer = SGD(
    model.parameters(),
    lr=BASE_LR * BATCH_SIZE / 256,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=NESTEROV,
)

scheduler = CosineAnnealingLR(optimizer, T_max=N_ANNEAL_EPOCHS)

###################### TRAINING LOOP #########################
device = "cuda"

def dataset_to_X_y(dataset):
    X = []
    y = []
    Z = []

    for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1024)):
        images, labels = batch

        h, z = model(images.to(device))

        X.append(h.cpu().numpy())
        Z.append(z.cpu().numpy())
        y.append(labels)

    X = np.vstack(X)
    Z = np.vstack(Z)
    y = np.hstack(y)

    return X, y, Z


history_type = []
epoch_hist = []
batch_hist = []
value_hist = []

def update_histories(type, epoch, batch, value):
    history_type.append(type)
    epoch_hist.append(epoch)
    batch_hist.append(batch)
    value_hist.append(value)

self_sim_momentum = 0.9
self_sim = torch.ones(len(cifar10_train), requires_grad=False)

def train_loop(model, n_epochs=N_ANNEAL_EPOCHS):
    model.to(device)
    model.train()
    training_start_time = time.time()

    update_histories("hyper_params", 0, -1, hyper_param_dict)
    
    # with torch.no_grad():
    #     X_train, y_train, Z_train = dataset_to_X_y(cifar10_train)
    # update_histories("embed_norm", 0, -1, np.linalg.norm(Z_train, axis=-1))
    # update_histories("embed", 0, -1, Z_train)


    for epoch_idx in range(n_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(cifar10_loader_ssl):

            print(f"batch: {batch_idx+1} / {len(cifar10_loader_ssl)}", end="\r")
            idx, og_image, label, view1, view2 = batch
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()

            _, z1 = model(view1)
            _, z2 = model(view2)
            z1.retain_grad()
            z2.retain_grad()
            z1_nld = F.normalize(z1)
            z2_nld = F.normalize(z2)

            loss = losses.nt_xent(
                z1,z2,
                attraction_factor=ATTRACTION_FACTOR,
                repulsion_factor=REPULSION_FACTOR
                )

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

            ##### RECORDING various histories
            with torch.no_grad():
                
                view1_grad1_cos_sim = (z1_nld * F.normalize(z1.grad)).sum(dim=-1).detach().cpu()
                view2_grad2_cos_sim = (z2_nld * F.normalize(z2.grad)).sum(dim=-1).detach().cpu()
                update_histories("view1_grad1_cos_sim", epoch_idx, batch_idx, view1_grad1_cos_sim)
                update_histories("view2_grad2_cos_sim", epoch_idx, batch_idx, view2_grad2_cos_sim)
                og_view_nld = F.normalize(torch.tensor(Z_train[idx], device=device))
                og_view_grad1_cos_sim = (og_view_nld * F.normalize(z1.grad)).sum(dim=-1).detach().cpu()
                og_view_grad2_cos_sim = (og_view_nld * F.normalize(z2.grad)).sum(dim=-1).detach().cpu()
                update_histories("og_view_grad1_cos_sim", epoch_idx, batch_idx, og_view_grad1_cos_sim)
                update_histories("og_view_grad2_cos_sim", epoch_idx, batch_idx, og_view_grad2_cos_sim)

                update_histories("view1and2_norm_before_optim_step", epoch_idx, batch_idx, torch.stack([z1.norm(dim=-1).detach(), z2.norm(dim=-1).detach()]).cpu())
                update_histories("batch_used_indicies", epoch_idx, batch_idx, idx)
                cos_sim_before_step = (z1_nld * z2_nld).sum(dim=-1).cpu().detach()
                update_histories("views_cos_sim_before_step", epoch_idx, batch_idx, cos_sim_before_step)
                cos_sim_to_og_view1 = (z1_nld * og_view_nld).sum(dim=-1).cpu().detach()
                cos_sim_to_og_view2 = (z2_nld * og_view_nld).sum(dim=-1).cpu().detach()
                update_histories("cos_sim_to_og_view_before_step", epoch_idx, batch_idx, torch.stack([cos_sim_to_og_view1, cos_sim_to_og_view2]))
                
                self_sim[idx] = (1-self_sim_momentum) * cos_sim_before_step + self_sim_momentum * self_sim[idx]

                _, z1_new = model(view1)
                _, z2_new = model(view2)
                z1_new_nld = F.normalize(z1_new)
                z2_new_nld = F.normalize(z2_new)

                cos_sim_after_step = (z1_new_nld * z2_new_nld).sum(dim=-1).cpu().detach()
                update_histories("views_cos_sim_after_step", epoch_idx, batch_idx, cos_sim_after_step)
                update1_nld = F.normalize(z1_new - z1)
                update2_nld = F.normalize(z2_new - z2)
                embed_update_cos_sim1 = (update1_nld * z1_nld).sum(dim=-1).cpu()
                embed_update_cos_sim2 = (update2_nld * z2_nld).sum(dim=-1).cpu()
                update_histories("embed_update_cos_sim", epoch_idx, batch_idx, torch.stack([embed_update_cos_sim1, embed_update_cos_sim2]).cpu().detach())
                update_histories("view1and2_norm_after_optim_step", epoch_idx, batch_idx, torch.stack([z1_new.norm(dim=-1), z2_new.norm(dim=-1)]).cpu())


                ##### Update gradient alignment
                update1_nld = F.normalize(z1_new - z1)
                update2_nld = F.normalize(z2_new - z2)
                update1_grad1_cos_sim = (update1_nld * F.normalize(z1.grad)).sum(dim=-1).cpu().detach()
                update2_grad2_cos_sim = (update2_nld * F.normalize(z2.grad)).sum(dim=-1).cpu().detach()
                update_histories("update1_grad1_cos_sim", epoch_idx, batch_idx, update1_grad1_cos_sim)
                update_histories("update2_grad2_cos_sim", epoch_idx, batch_idx, update2_grad2_cos_sim)
                ##### 

                cos_sim_to_og_view1_new = (z1_new_nld * og_view_nld).sum(dim=-1).cpu()
                cos_sim_to_og_view2_new = (z2_new_nld * og_view_nld).sum(dim=-1).cpu()
                update_histories("cos_sim_to_og_view_after_step", epoch_idx, batch_idx, torch.stack([cos_sim_to_og_view1_new, cos_sim_to_og_view2_new]))


        end_time = time.time()
        if (epoch_idx + 1) % PRINT_EVERY_EPOCHS == 0:
            print(
                f"Epoch {epoch_idx + 1}, "
                f"average loss {epoch_loss / len(cifar10_loader_ssl):.4f}, "
                f"{end_time - start_time:.1f} s",
                flush=True
            )
            print("self sim mean:",self_sim.mean().item())
            with torch.no_grad():
            #### dataset level histories
                model.eval()  
                X_train, y_train, Z_train = dataset_to_X_y(cifar10_train)
                X_test, y_test, Z_test = dataset_to_X_y(cifar10_test)
                model.train()
                knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
                knn.fit(X_train, y_train)
                score = knn.score(X_test, y_test)
                print(f"kNN accuracy (cosine): {score:.4f}", flush=True)
                
                update_histories("kNN_accuracy_cosine", epoch_idx, batch_idx, score)
                # update_histories("embed", epoch_idx, batch_idx, Z_train)
                update_histories("embed_norm", epoch_idx, batch_idx, np.linalg.norm(Z_train, axis=-1))
            
            
            embed_histories_df = pd.DataFrame.from_dict(
                dict(
                    type = history_type,
                    epoch = epoch_hist,
                    batch = batch_hist,
                    value = value_hist
                    )
                )
            embed_histories_df.to_pickle(embed_histories_path)

        scheduler.step()

    training_end_time = time.time()
    hours = (training_end_time - training_start_time) / 60 // 60
    minutes = (training_end_time - training_start_time) / 60 % 60
    print(
        f"Total training length for {n_epochs} epochs: {hours:.0f}h {minutes:.0f}min",
        flush=True
    )

train_loop(model, N_EPOCHS)

###################### FINAL EVALUATION #########################


model.eval()
with torch.no_grad():
    X_train, y_train, Z_train = dataset_to_X_y(cifar10_train)
    X_test, y_test, Z_test = dataset_to_X_y(cifar10_test)

knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
knn.fit(X_train, y_train)
print(f"kNN accuracy (cosine): {knn.score(X_test, y_test):.4f}", flush=True)


embed_histories_df = pd.DataFrame.from_dict(
     dict(
         type = history_type,
         epoch = epoch_hist,
         batch = batch_hist,
         value = value_hist
        )
     )
embed_histories_df.to_pickle(embed_histories_path)
