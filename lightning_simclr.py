import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import numpy as np
import time
import os
from inspect import getfullargspec

from sklearn.neighbors import KNeighborsClassifier
from norm_functions import *
from data_utils import get_train_and_test_set, get_dataset, get_augmented_dataloader

###################### PARAMS ##############################

NORM_FUNCTION_DICT = {
                        'l2': F.normalize, # if no norm_function_str is provided this will be used (default SimCLR)
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        "minus_exp_map": minus_exp_map_normalize,
                        'stereo': stereo_normalize,
                        "stereo_-_mean": stereo_minus_mean_normalize,
                        "stereo_div_mean": stereo_div_mean_normalize,
                        "mono": mono_normalize,
                        "line": line_normalize,
                        "exp2": exp2_normalize,
                        "exponent": exponent_normalize,
                        "torus": torus_norm,
                    }



###################### NETWORK ARCHITECTURE #########################

class ResNet18withProjector(nn.Module):
    def __init__(self, dataset_name, projector_hidden_size=1024):
        super().__init__()

        self.backbone = resnet18(weights=None)

        if "cifar" in dataset_name:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()
        
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, projector_hidden_size), 
            nn.ReLU(), 
            nn.Linear(projector_hidden_size, 128),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def infoNCE(
        features, 
        temperature=0.5, 
        norm_function=F.normalize,
        power = 0
        ):
    batch_size = features.size(0) // 2

    x = norm_function(features)
    RescaleNorm.apply(x, power)
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

###################### ANDREW'S MODIFICATIONS #########################

class RescaleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, power=0):
        ctx.save_for_backward(z)
        ctx.power = power
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        power = ctx.power
        norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)

        return grad_output * norm**power, None

@torch.no_grad()
def cut_weights(model, cut_ratio):
    for parameter in model.parameters():
        parameter.data = parameter.data / cut_ratio


###################### LIGHTNING MODULE #########################

class SimCLR(L.LightningModule):

    def __init__(
            self,
            dataset_name,
            base_lr = 0.06,
            batch_size = 1024,
            weight_decay = 5e-4,
            nestrov = False,
            n_epochs = 1000,
            norm_function_str = "l2",
            norm_scaling = 1,
            warmup_epochs = 1,
            log_acc_every = 10,
            cut = 1,
            exponent = 1,
            power = 0,
            seed = get_truly_random_seed_through_os()
                 ):
        super().__init__()

        ignore_list = ["log_acc_every"]
        self.save_hyperparameters(ignore=ignore_list)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ResNet18withProjector(dataset_name)
        self.cut = cut
        cut_weights(self.model, cut)

        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]
        self.norm_scaling = norm_scaling
        self.exponent = exponent
        if "norm_scaling" in getfullargspec(self.norm_function).args:
            self.loss = lambda features: infoNCE(features, norm_function=lambda norm: self.norm_function(norm, norm_scaling=self.norm_scaling))
        if "exponent" in getfullargspec(self.norm_function).args:
            self.loss = lambda features: infoNCE(features, norm_function=lambda norm: self.norm_function(norm, exponent=self.exponent))
        else:
            self.loss = lambda features: infoNCE(features, norm_function=lambda norm: self.norm_function(norm), power=power)

        self.base_lr = base_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.nestrov = nestrov
        self.n_epochs = n_epochs
        self.norm_function_str = norm_function_str
        self.warmup_epochs = warmup_epochs
        self.log_acc_every = log_acc_every
        #self.seed = seed
        self.dataset_name = dataset_name

        dataset = get_dataset(dataset_name)
        self.knn_train_dataset, self.knn_test_dataset = get_train_and_test_set(dataset)

        self.previous_embeds = None
        self.embed_distances_history = []
        self.norm_history = []
    
    # def __getattribute__(self, name):
    #     # This allows for the hparams to also be accessed via self.hp_name
    #     try:
    #         return super.__getattribute__(name)
    #     except:
    #         return self.hparams[name]
    #     if name in self.hparams.keys():
    #         return self.hparams[name]
    #     else:
    #         # Default behaviour
    #         return self.__getattribute__(name)
    
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

        if len(view1) == self.batch_size: # only log batches with full size
            self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

###################### EVALUATION #########################

    def embed_dataset(self, dataset):
        with torch.no_grad():
            X, y, Z = [], [], []

            for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1024)):
                images, labels = batch

                h, z = self.model(images.to(self.device))

                X.append(h.cpu().numpy())
                Z.append(z.cpu().numpy())
                y.append(labels)

            X = np.vstack(X)
            Z = np.vstack(Z)
            y = np.hstack(y)

            return X, y, Z

    def log_embed_histories(self, projector_embeddings_np):
        projector_embeddings = torch.tensor(projector_embeddings_np)
        with torch.no_grad():
            if self.previous_embeds is None:
                    self.previous_embeds = self.norm_function(projector_embeddings).cpu().numpy()
            else:
                embeddings_on_sphere = self.norm_function(projector_embeddings).cpu().numpy()
                new_distances = np.linalg.norm(self.previous_embeds - embeddings_on_sphere, axis=1)
                new_norms = np.linalg.norm(projector_embeddings, axis=1)
                self.norm_history.append(new_norms)
                self.embed_distances_history.append(new_distances)
                self.previous_embeds = projector_embeddings

    
    def get_knn_acc(self, train_dataset=None, test_dataset=None, n_neighbors=10):
        if not train_dataset:
            train_dataset = self.knn_train_dataset
        if not test_dataset:
            test_dataset = self.knn_test_dataset
        self.model.eval()
        
        X_train, y_train, Z_train = self.embed_dataset(train_dataset)
        X_test, y_test, Z_test = self.embed_dataset(test_dataset)
        
        self.log_embed_histories(Z_train)
        if self.current_epoch > 0:
            with open(f"{self.logger.log_dir}/embed_history.npy", "wb") as file:
                np.save(
                    file,
                    dict(
                        distance_history = np.vstack(self.embed_distances_history),
                        norm_history = np.vstack(self.norm_history)
                    )
                )

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        
        self.model.train()
        return score

    def on_fit_start(self):
        knn_acc = self.get_knn_acc()
        self.logger.log_hyperparams(self.hparams, {"kNN accuracy (cosine)": knn_acc})
        self.logger.experiment.add_scalar("kNN accuracy (cosine)", knn_acc)
        return super().on_fit_start()
    
    def on_train_epoch_end(self):
        cur_epoch = self.current_epoch + 1
        if cur_epoch % self.log_acc_every == 0 or cur_epoch == self.n_epochs:
            knn_acc = self.get_knn_acc()
            self.log("kNN accuracy (cosine)", knn_acc)
        else:
            with torch.no_grad():
                X_train, y_train, Z_train = self.embed_dataset(self.knn_train_dataset)
            self.log_embed_histories(Z_train)


###################### ACTUAL TRAINING #########################

log_dir = "logs"
checkpoint_every = 100

def train_variants(variants, experiment_set_name, dataset_name):

    for variant in variants:
        
        mod = SimCLR(dataset_name=dataset_name, **variant)
        variant_name = variant.pop("name", mod.norm_function_str)
        print(mod.hparams)
        logger = TensorBoardLogger(
            log_dir,
            name = f"{experiment_set_name}/{variant_name}"
            )
        # saving a checkpoint at "checkpoint_every" epochs - no overwrite
        checkpoint_callback = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}",
                every_n_epochs=checkpoint_every,
                save_top_k=-1,
            )
        # saving the newest model - overwrites previous newest
        checkpoint_callback2 = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}{step:02d}",
                every_n_epochs=1,
                save_top_k=1,
            )
        
        trainer = L.Trainer(
            callbacks=[checkpoint_callback, checkpoint_callback2],
            max_epochs=mod.n_epochs,
            logger=logger)
        data_loader = get_augmented_dataloader(mod.dataset_name)
        trainer.fit(mod, data_loader)

import numpy as np

variants = sum([[
    dict(norm_function_str = "l2", seed=seed),
] for seed in range(3)], [])

if __name__ == "__main__":
    experiment_set_name = "Point_tracking-1000"
    dataset_name = "cifar10"
    train_variants(variants, experiment_set_name, dataset_name)
    
# CUDA_VISIBLE_DEVICES=2 python -i lightning_simclr_cifar10.py