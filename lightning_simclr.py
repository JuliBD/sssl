import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, SequentialLR, LinearLR, ConstantLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import numpy as np
from typing import Literal
from sklearn.neighbors import KNeighborsClassifier
from utils.norm_functions import *
from utils.data_utils import *


###################### PARAMS ##############################

NORM_FUNCTION_DICT = {
                        'l2': F.normalize, # if no norm_function_str is provided this will be used (same as in default SimCLR)
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        'stereo': stereo_normalize,
                        "mono": mono_normalize,
                        "torus": torus_norm,
                    }


###################### NETWORK ARCHITECTURE #########################

class ResNet18withProjector(nn.Module):
    def __init__(
            self, 
            dataset_name, 
            projector_hidden_size=1024,
            embedding_size=128,
            use_ln = False
            ):
        super().__init__()
        self.use_ln = use_ln

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
            nn.Linear(projector_hidden_size, embedding_size),
        )
        
        if use_ln:
            self.ln = nn.LayerNorm(embedding_size)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)

        if self.use_ln:
            z = self.ln(z)

        return h, z
        
###################### LOSS FUNCTION #########################
def infoNCE(
        features,
        temperature=0.5, 
        norm_function=F.normalize,
        ):
    batch_size = features.size(0) // 2
    x = norm_function(features)
    cos_xx = x @ x.T / temperature

    cos_xx.fill_diagonal_(float("-inf"))
    
    targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)


##################### GRADIENT  MODIFICATIONS #########################

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


class RotateGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, strength):
        ctx.save_for_backward(z)
        ctx.strength = strength
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        strength = ctx.strength
        # rotate gradient to point towards the hypersphere
        with torch.no_grad():
            new_grad_output = F.normalize(z - F.normalize(z, dim=-1), dim=-1)
            new_grad_output_rescaled = new_grad_output * grad_output.norm(dim=-1, keepdim=True)
            biased_grad = grad_output + strength * new_grad_output_rescaled
        
        return biased_grad , None


###################### LIGHTNING MODULE #########################

class SimCLR(L.LightningModule):

    def __init__(
            self,
            dataset_name: Literal["cifar10",
                                "cifar10_unbalanced" ,
                                "cifar100" ,
                                "cifar100_unbalanced",
                                "flowers" ],
            base_lr = 0.06,
            batch_size = 1024,
            weight_decay = 5e-4,
            momentum = 0.9,
            nestrov = False,
            n_epochs = 1000,
            norm_function_str = "l2",
            warmup_epochs = 1,
            log_acc_every = 10,
            record_embed_histories = True,
            cut = 1,
            power = 0,
            seed = None,
            use_lr_schedule = True,
            rotate_factor = 0.0,
            optimizer = "sgd",
            use_ln = False,
            adjust_head = False,
                 ):
        super().__init__()

        hp_ignore_list = ["log_acc_every", "record_embed_histories"]
        if seed is None: seed = get_truly_random_seed_through_os()
        self.save_hyperparameters(ignore=hp_ignore_list)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ResNet18withProjector(dataset_name, use_ln=use_ln)
        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]

         # Cut initialization
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.data = parameter.data / cut
        
        if adjust_head:
            with torch.no_grad():
                for parameter in self.model.projector[-1].parameters():
                    parameter.data = parameter.data * (base_lr / 0.06)
    
        self.loss = infoNCE


        self.idx_log_temp = []
        self.grad_norm_log_temp = []
        self.grad_norm_log = []
        self.train_norm_log_temp = []
        self.train_norm_log = []
        
        dataset = get_dataset(dataset_name)
        self.knn_train_dataset, self.knn_test_dataset = get_train_and_test_set(dataset)

        # saving hyperparameters for later access
        self.use_ln = use_ln
        self.record_embed_histories = record_embed_histories
        self.cut = cut
        self.power = power
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nestrov = nestrov
        self.n_epochs = n_epochs
        self.norm_function_str = norm_function_str
        self.warmup_epochs = warmup_epochs
        self.log_acc_every = log_acc_every
        self.seed = seed
        self.dataset_name = dataset_name
        self.use_lr_schedule = use_lr_schedule
        self.rotate_factor = rotate_factor
        self.optimizer = optimizer

    
    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.base_lr * self.batch_size / 256,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nestrov,
                fused=True
            )
        elif self.optimizer == "adamw":
            # Adam works almost as well but requires 1e-6 weight decay and warmup 10 an lr 3e-3
            optimizer = AdamW(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
                fused=True
            )
        
        if self.use_lr_schedule:

            lr_scheduler = {
                'scheduler': SequentialLR(
                        optimizer,
                        schedulers=[
                            LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_epochs),
                            CosineAnnealingLR(optimizer, T_max=self.n_epochs - self.warmup_epochs),
                        ],
                        milestones=[self.warmup_epochs],
                    ),
                "name": 'schedule'
            }
        else:
            lr_scheduler = {
                'scheduler': ConstantLR(optimizer, factor=1, total_iters=self.n_epochs),
                "name": 'schedule'
            }
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch):
        idx, view1, view2, _ = batch
        
        _, z1 = self.model(view1)
        _, z2 = self.model(view2)

        features = torch.cat((z1, z2))
        features = RescaleNorm.apply(features, self.power)
        features = RotateGrad.apply(features, self.rotate_factor)

        loss = self.loss(features, norm_function=self.norm_function)

        if len(view1) == self.batch_size: # only log batches with full size
            self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.record_embed_histories:
            # logging embedding norms
            self.idx_log_temp.append(idx.detach().cpu())
            self.train_norm_log_temp.append(
                torch.stack(
                    [z1.norm(dim=-1).detach().cpu(),
                    z1.norm(dim=-1).detach().cpu()]
                    )
                )

            # retaining gradients for gradient norm logging
            z1.retain_grad()
            z2.retain_grad()
            self.last_z1 = z1
            self.last_z2 = z2

        return loss
    
    def on_after_backward(self): # logging gradient norms
        if self.model is not None and self.model.training and self.record_embed_histories and self.last_z1 is not None and self.last_z1.grad is not None:
            self.grad_norm_log_temp.append(
                torch.stack(
                    [self.last_z1.grad.norm(dim=-1).detach().cpu(),
                    self.last_z2.grad.norm(dim=-1).detach().cpu()]
                    )
                )
###################### EVALUATION #########################

    def embed_dataset(self, dataset):
        with torch.no_grad():
            X, y, Z = [], [], []

            for batch in DataLoader(dataset, batch_size=1024):
                images, labels = batch

                h, z = self.model(images.to(self.device))

                X.append(h.cpu().numpy())
                Z.append(z.cpu().numpy())
                y.append(labels)

            X = np.vstack(X)
            y = np.hstack(y)
            Z = np.vstack(Z)

            return X, y, Z

    def log_embed_histories(self):
        if not self.record_embed_histories: return None
        
        if len(self.idx_log_temp) > 1:
            sort_idx = torch.cat(self.idx_log_temp).argsort()
            self.train_norm_log.append(torch.cat(self.train_norm_log_temp, dim=-1)[:,sort_idx])
            self.grad_norm_log.append(torch.cat(self.grad_norm_log_temp, dim=-1)[:,sort_idx])
            self.idx_log_temp = []
            self.train_norm_log_temp = []
            self.grad_norm_log_temp = []
    
    def save_histories(self):
        with open(f"{self.logger.log_dir}/embed_history.npy", "wb") as file:
                    np.save(
                        file,
                        dict(
                            train_norm_history = np.stack(self.train_norm_log),
                            grad_norm_history = np.stack(self.grad_norm_log),
                        )
                    )
            
    def get_knn_acc_on_dataset(self, train_dataset, test_dataset, n_neighbors=10):
        
        self.model.eval()
        train_embeds = self.embed_dataset(train_dataset)
        test_embeds = self.embed_dataset(test_dataset)
        self.model.train()

        return self.knn_acc(train_embeds, test_embeds, n_neighbors)
    
    def knn_acc(self, train_embeds, test_embeds, n_neighbors=10):
        
        X_train, y_train, Z_train = train_embeds
        X_test, y_test, Z_test = test_embeds

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        
        return score

    def on_fit_start(self):
        train_embeds = self.embed_dataset(self.knn_train_dataset)
        test_embeds = self.embed_dataset(self.knn_test_dataset)

        self.log_embed_histories()
        knn_acc = self.knn_acc(train_embeds, test_embeds)

        self.logger.log_hyperparams(self.hparams, {"kNN accuracy (cosine)": knn_acc})
        return super().on_fit_start()
    
    def on_train_epoch_end(self):
        self.model.eval()

        cur_epoch = self.current_epoch + 1

        self.log_embed_histories()
        mean = self.train_norm_log[-1].mean().item()
        std = self.train_norm_log[-1].std().item()
        self.log("Mean embedding norm", mean)
        self.log("Std: embedding norm", std)
        self.log("Std / Mean", (std)/(mean))

        if cur_epoch % self.log_acc_every == 0 or cur_epoch == self.n_epochs:
            train_embeds = self.embed_dataset(self.knn_train_dataset)
            test_embeds = self.embed_dataset(self.knn_test_dataset)
            knn_acc = self.knn_acc(train_embeds, test_embeds)
            self.log("kNN accuracy (cosine)", knn_acc)
            self.save_histories()
                
        self.model.train()


###################### ACTUAL TRAINING #########################

log_dir = "logs"
checkpoint_every = 100

def train_variants(variants, experiment_set_name, dataset_name):

    for variant in variants:
        
        variant_name = variant.pop("name", variant["norm_function_str"])
        variant_dataset = variant.pop("dataset_name", dataset_name)
        variant_knn_dataset = variant.pop("knn_dataset", None)
        
        mod = SimCLR(dataset_name=variant_dataset, **variant)
        if variant_knn_dataset:
            mod.knn_train_dataset, mod.knn_test_dataset = get_train_and_test_set(get_dataset(variant_knn_dataset))
        print(mod.hparams)

        logger = TensorBoardLogger(
            log_dir,
            name = f"{experiment_set_name}/{variant_name}"
            )
        # saving a checkpoint at "checkpoint_every" epochs - no overwrite
        permanent_checkpoints = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}",
                every_n_epochs=checkpoint_every,
                save_top_k=-1,
            )
        # saving the newest model - overwrites previous newest
        newest_checkpoint = ModelCheckpoint(
                dirpath=logger.log_dir+"/checkpoint/",
                filename="{epoch:02d}{step:02d}",
                every_n_epochs=1,
                save_top_k=1,
            )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        trainer = L.Trainer(
            callbacks=[permanent_checkpoints, newest_checkpoint, lr_monitor],
            max_epochs=mod.n_epochs,
            logger=logger)
        data_loader = get_augmented_dataloader(mod.dataset_name)
        trainer.fit(mod, data_loader)

variants = sum([[
    # dict(name="angle", norm_function_str = "angle", seed=seed, warmup_epochs=1),
    # dict(name="angle2", norm_function_str = "angle2", seed=seed, warmup_epochs=1),
    # dict(name="stereo", norm_function_str = "stereo", seed=seed, warmup_epochs=1),
    # dict(name="mono", norm_function_str = "mono", seed=seed, warmup_epochs=1),


    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*2, weight_decay=5e-4/2, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*4, weight_decay=5e-4/4, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*8, weight_decay=5e-4/8, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*32, weight_decay=5e-4/32, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*128, weight_decay=5e-4/128, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*256, weight_decay=5e-4/256, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*1024, weight_decay=5e-4/1024, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06*2048, weight_decay=5e-4/2048, n_epochs=100),

    dict(name="lr_weight_decay_rebalance", norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06, weight_decay=5e-4, n_epochs=100),

    dict(name="andrews_variants/gradscale", norm_function_str = "l2", seed=seed, warmup_epochs=100, base_lr=0.06/6, power=1),
    dict(name="mappings_onto_hypersphere/exp_map", norm_function_str = "exp_map", seed=seed, warmup_epochs=1),

    # dict(name="SimCLR", norm_function_str = "l2", seed=seed, warmup_epochs=1),
    # dict(name="rotate_grad", norm_function_str = "l2", seed=seed, warmup_epochs=1, rotate_factor=0.01),
    # dict(name="SimCLR_adam", optimizer="adamw", norm_function_str = "l2", seed=seed, warmup_epochs=10, base_lr=3e-3, weight_decay=1e-6),
    # dict(name="pre-ln", norm_function_str = "l2", seed=seed, warmup_epochs=1),
    # dict(name="ln-sdg-rebalance", norm_function_str = "l2", seed=seed, warmup_epochs=1, base_lr=0.06 * fact, weight_decay=5e-4/fact),
    # dict(name="grad_scale_ln", norm_function_str = "l2", seed=seed, warmup_epochs=1, power=1, use_ln=True),
    # dict(name="cut", norm_function_str = "l2", cut=3, seed=seed, warmup_epochs=1),

    # dict(name="SimCLR-cifar100", dataset_name = "cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1),
    # dict(name="rotate_grad-cifar100", dataset_name = "cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, rotate_factor=0.01),
    # dict(name="SimCLR_ln-cifar100", dataset_name = "cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, use_ln=True),

    # dict(name="gradscle_ln-cifar100_unba", dataset_name = "cifar100_unbalanced", norm_function_str = "l2", seed=seed, warmup_epochs=1, use_ln=True, power=1, n_epochs=1000),
    # dict(name="cut-cifar100_unba", dataset_name = "cifar100_unbalanced", norm_function_str = "l2", seed=seed, warmup_epochs=1, cut=3, n_epochs=1000),

    # dict(name="gradscle_ln-cifar100_unba", dataset_name = "cifar100_unbalanced", knn_dataset="cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, use_ln=True, power=1, n_epochs=3333),
    # dict(name="cut-cifar100_unba", dataset_name = "cifar100_unbalanced", knn_dataset="cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, cut=3, n_epochs=3333),
    
    # dict(name="SimCLR-cifar100_unba", dataset_name = "cifar100_unbalanced", knn_dataset="cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, n_epochs=3333),
    # dict(name="rotate_grad-cifar100_unba", dataset_name = "cifar100_unbalanced", knn_dataset="cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, rotate_factor=0.01, n_epochs=3333),
    # dict(name="SimCLR_ln-cifar100_unba", dataset_name = "cifar100_unbalanced", knn_dataset="cifar100", norm_function_str = "l2", seed=seed, warmup_epochs=1, use_ln=True, n_epochs=3333),

    # dict(name="adam_testing", norm_function_str = "l2", base_lr=3e-3 / 10, weight_decay=1e-6 *10, seed=seed, warmup_epochs=10),
    # dict(name="my_adam_fixed_var", optimizer="myadamw", norm_function_str = "l2", base_lr=3e-3, weight_decay=1e-6, seed=seed, warmup_epochs=10, n_epochs=100),
    # dict(name="adam", optimizer="adamw", norm_function_str = "l2", base_lr=3e-3, weight_decay=1e-6, seed=seed, warmup_epochs=10, n_epochs=100),
    # dict(name="sgd", optimizer="sgd", norm_function_str = "l2", seed=seed, warmup_epochs=1, n_epochs=100),
    # dict(name="no_schedule", norm_function_str = "l2", seed=seed, use_lr_schedule=False),
] for seed in range(1,4)], [])

if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`*")

    experiment_set_name = "reruns"
    dataset_name = "cifar10"
    train_variants(variants, experiment_set_name, dataset_name)
    
# CUDA_VISIBLE_DEVICES=2 python lightning_simclr.py