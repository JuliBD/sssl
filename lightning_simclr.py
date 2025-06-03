import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import time
from torchvision.datasets import CIFAR10, CIFAR100

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from norm_functions import *
from data_utils import get_train_and_test_set, get_dataset, get_augmented_dataloader
from data_utils import BATCH_SIZE, cifar10_test, cifar10_train, cifar10_loader_ssl

###################### PARAMS ##############################

N_EPOCHS = 1000
BASE_LR = 0.06
WEIGHT_DECAY = 5e-4
PROJECTOR_HIDDEN_SIZE = 1024
NORM_FUNCTION_STR = "l2"
WARMUP_EPOCHS = 1
LOG_ACC_EVERY = 10
CLAMP = False
NESTEROV = False
NORM_SCALING = 1


NORM_FUNCTION_DICT = {
                        'l2': F.normalize, # if no norm_function_str is provided this will be used (default SimCLR)
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        'stereo': stereo_normalize,
                        "mono": mono_normalize,
                        "torus": torus_norm,
                    }



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
        self.warmup_epochs = warmup_epochs + 1
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
        

###################### EVALUATION #########################


def dataset_to_X_y(dataset, model, device):
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


def get_knn_acc(
        model,
        device,
        train_dataset=cifar10_train,
        test_dataset=cifar10_test
        ):
    model.eval()
    
    with torch.no_grad():
        X_train, y_train, Z_train = dataset_to_X_y(train_dataset, model, device)
        X_test, y_test, Z_test = dataset_to_X_y(test_dataset, model, device)

    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    
    model.train()

    return score

###################### LIGHTNING MODULE #########################

class SimCLR(L.LightningModule):

    def __init__(
            self,
            dataset_name = "cifar10",
            base_lr = BASE_LR,
            batch_size = BATCH_SIZE,
            weight_decay = WEIGHT_DECAY,
            nestrov = NESTEROV,
            n_epochs = N_EPOCHS,
            norm_function_str = NORM_FUNCTION_STR,
            norm_scaling = NORM_SCALING,
            warmup_epochs = WARMUP_EPOCHS,
            log_acc_every = LOG_ACC_EVERY,
                 ):
        super().__init__()

        ignore_list = ["log_acc_every"]
        self.save_hyperparameters(ignore=ignore_list)
        print(self.hparams)

        self.model = ResNet18withProjector()
        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]
        self.norm_scaling = norm_scaling
        try:
            self.loss = lambda features: infoNCE(features, norm_function=lambda norm: self.norm_function(norm, 1, self.norm_scaling))
        except:
            self.loss = lambda features: infoNCE(features, norm_function=lambda norm: self.norm_function(norm))

        self.base_lr = base_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.nestrov = nestrov
        self.n_epochs = n_epochs
        self.norm_function_str = norm_function_str
        self.warmup_epochs = warmup_epochs
        self.log_acc_every = log_acc_every
        dataset = get_dataset(dataset_name)
        self.train_dataset, self.test_dataset = get_train_and_test_set(dataset)
    
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
        
        scheduler = CosineAnnealingWarmup(
            optimizer,
            T_max=self.n_epochs,
            warmup_epochs=self.warmup_epochs
            )
        return [optimizer], [scheduler]
    
    def training_step(self, batch):
        view1, view2, _ = batch
        view1 = view1
        view2 = view2

        _, z1 = self.model(view1)
        _, z2 = self.model(view2)

        loss = self.loss(torch.cat((z1, z2)))

        if len(view1) == self.batch_size: # only record batches with full size
            self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_fit_start(self):
        knn_acc = get_knn_acc(self.model, self.device)
        self.logger.log_hyperparams(self.hparams, {"kNN accuracy (cosine)": knn_acc})
        self.logger.experiment.add_scalar("kNN accuracy (cosine)", knn_acc)
        return super().on_fit_start()
    
    def on_train_epoch_end(self):
        cur_epoch = self.current_epoch + 1
        if cur_epoch % self.log_acc_every == 0 or cur_epoch == self.n_epochs:
            knn_acc = get_knn_acc(self.model, self.device)
            self.log("kNN accuracy (cosine)", knn_acc)


###################### ACTUAL COMPUTING #########################


from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
log_dir = "logs"
checkpoint_every = 100

def train_variants(variants, experiment_set_name = "SimCLR-"):

    for variant in variants:
        
        variant_name = variant.pop("name", experiment_set_name)
        dataset_name = variant.pop("dataset_name", "cifar10_unbalanced")
        pl_module = SimCLR(dataset_name=dataset_name, **variant)
        logger = TensorBoardLogger(
            log_dir,
            name = f"{variant_name}/{pl_module.norm_function_str}"
            )
        # saving a checkpoint every "checkpoint_every" epochs - no overwrite
        checkpoint_callback = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}",
                every_n_epochs=checkpoint_every,
                save_top_k=-1,
            )
        # saving the newest model, overwrites previous newest
        checkpoint_callback2 = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}{step:02d}",
                every_n_epochs=1,
                save_top_k=1,
            )
        
        n_epochs = variant.get("n_epochs", N_EPOCHS)
        trainer = L.Trainer(
            callbacks=[checkpoint_callback, checkpoint_callback2],
            max_epochs=n_epochs,
            logger=logger)
        data_loader = get_augmented_dataloader(dataset_name)
        trainer.fit(pl_module, data_loader)

import numpy as np
# variants = [
#     # dict(norm_function_str = "stereo", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=1/10),
#     # dict(norm_function_str = "stereo", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=10),
#     # dict(norm_function_str = "stereo", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=0.64),
#     # dict(norm_function_str = "stereo", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=1/0.64),
#     dict(norm_function_str = "exp_map", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=torch.pi/2), # this scaling and norm_function causes the loss to become nan
#     dict(norm_function_str = "exp_map", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=0.64),
#     dict(norm_function_str = "angle2", base_lr = 0.06, warmup_epochs=0, n_epochs=100),
#     dict(norm_function_str = "l2", base_lr = 0.06, warmup_epochs=0, n_epochs=100),
# ]
variants = [dict(norm_function_str = "l2", base_lr = 0.06, warmup_epochs=1, n_epochs=100),]
# [
#   dict(norm_function_str = "stereo", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=x) for x in np.linspace(0.3,0.8, 10)
# ] +\
# [
#   dict(norm_function_str = "exp_map", base_lr = 0.06, warmup_epochs=1, n_epochs=100, norm_scaling=x) for x in np.linspace(0.3, 0.8, 10)
# ] 


if __name__ == "__main__":
    experiment_set_name = "Unbalanced3"
    train_variants(variants, experiment_set_name)
    
# CUDA_VISIBLE_DEVICES=2 python -i lightning_simclr_cifar10.py