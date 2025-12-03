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
                        'l2': F.normalize, # default
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        'stereo': stereo_normalize,
                        "mono": mono_normalize,
                        "torus": torus_norm,
                        "none": lambda x: x,
                    }


###################### NETWORK ARCHITECTURE #########################

class ResNet18withProjector(nn.Module):
    def __init__(
            self, 
            dataset_name, 
            projector_hidden_size=1024,
            embedding_size=128,
            ):
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
            nn.Linear(projector_hidden_size, embedding_size),
        )
        

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)

        return h, z
        
###################### LOSS FUNCTION #########################

def infoNCE(
        features,
        temperature=0.5, 
        norm_function=F.normalize,
        shuffle_positive_pairs = False,
        ):
    batch_size = features.size(0) // 2

    x = norm_function(features)
    cos_xx = x @ x.T / temperature

    cos_xx.fill_diagonal_(float("-inf"))
    
    targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    if shuffle_positive_pairs:
        perm = torch.randperm(batch_size, dtype=int, device=cos_xx.device)
        targets[:batch_size] = targets[:batch_size][perm]
        perm = torch.randperm(batch_size, dtype=int, device=cos_xx.device)
        targets[batch_size:] = targets[batch_size:][perm]

    return F.cross_entropy(cos_xx, targets)


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


###################### SIMCLR LIGHTNING MODULE #########################

class SimCLR(L.LightningModule):

    def __init__(
            self,
            dataset_name: Literal["cifar10",
                                "cifar10_unbalanced" ,
                                "cifar100" ,
                                "cifar100_unbalanced",
                                "flowers"],
            base_lr = 0.06,
            batch_size = 1024,
            weight_decay = 5e-4,
            momentum = 0.9,
            nestrov = False,
            n_epochs = 1000,
            norm_function_str = "l2",
            warmup_epochs = 1,
            log_acc_every = 10,
            log_embeds_flag = True,
            cut = 1,
            power = 0,
            seed = None,
            use_lr_schedule = True,
            optimizer = "sgd",
            adjust_head = False,
            shuffle_positive_pairs = False,
                 ):
        super().__init__()

        hp_ignore_list = ["log_acc_every", "log_embeds_flag"]
        self.save_hyperparameters(ignore=hp_ignore_list)

        if seed is None: seed = get_truly_random_seed_through_os()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = ResNet18withProjector(dataset_name)

        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]

         # Cut initialization
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.data = parameter.data / cut
        
        if adjust_head: # adjust the parameters of only the last layer
            with torch.no_grad():
                for parameter in self.model.projector[-1].parameters():
                    parameter.data = parameter.data * (base_lr / 0.06)
    
        self.idx_log_temp = []
        self.grad_norm_log_temp = []
        self.grad_norm_log = []
        self.train_norm_log_temp = []
        self.train_norm_log = []
        self.norm_change_log_temp = []
        self.norm_change_log = []

        self.norm_log_batch = []
        self.norm_change_log_batch = []
        self.moved_dist_log_batch = []
        
        dataset = get_dataset(dataset_name)
        self.knn_train_dataset, self.knn_test_dataset = get_train_and_test_set(dataset)

        # saving hyperparameters for later access
        self.log_embeds_flag = log_embeds_flag
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
        self.optimizer = optimizer
        self.shuffle_positive_pairs = shuffle_positive_pairs

    
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


###################### TRAINING STEP #########################
    
    def training_step(self, batch):
        idx, view1, view2, _ = batch
        
        _, z1 = self.model(view1)
        _, z2 = self.model(view2)

        features = torch.cat((z1, z2))
        features = RescaleNorm.apply(features, self.power)

        loss = infoNCE(features, norm_function=self.norm_function, shuffle_positive_pairs=self.shuffle_positive_pairs)

        if len(view1) == self.batch_size: # only log batches with full size
            self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.log_embeds_flag:
            # logging embedding norms
            self.idx_log_temp.append(idx.detach().cpu())

            # retaining gradients and save for gradient norm logging
            z1.retain_grad()
            z2.retain_grad()

            self.last_z1 = z1
            self.last_z2 = z2

            self.last_view1 = view1
            self.last_view2 = view2

        return loss

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

###################### LOGGING #########################
    def on_fit_start(self):
        train_embeds = self.embed_dataset(self.knn_train_dataset)
        test_embeds = self.embed_dataset(self.knn_test_dataset)

        knn_acc = self.knn_acc(train_embeds, test_embeds)

        self.logger.log_hyperparams(self.hparams, {"kNN accuracy (cosine)": knn_acc})
        return super().on_fit_start()
    
    def on_after_backward(self): # logging gradient norms and embedding norms
        if self.model.training\
            and self.log_embeds_flag\
            and self.last_z1 is not None\
            and self.last_z1.grad is not None:
            # and len(self.last_z1) == self.batch_size
            embed_norms = torch.stack(
                    [self.last_z1.norm(dim=-1),
                    self.last_z2.norm(dim=-1)]
                    ).detach().cpu()
            grad_norms = torch.stack(
                    [self.last_z1.grad.norm(dim=-1),
                    self.last_z2.grad.norm(dim=-1)]
                    ).detach().cpu()

            self.train_norm_log_temp.append(embed_norms)
            self.grad_norm_log_temp.append(grad_norms)
            self.norm_log_batch.append(embed_norms.mean().item())

            if len(self.last_z1) == self.batch_size:
                self.log("Embedding Norm - Batch", embed_norms.mean().item())
                self.log("Gradient Norm - Batch", grad_norms.mean().item())
            
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if self.model.training\
            and self.log_embeds_flag\
            and self.last_z1 is not None:
        
            with torch.no_grad():
                _, new_z1 = self.model(self.last_view1)
                _, new_z2 = self.model(self.last_view2)

            z1_embed_norm_diff = (new_z1.norm(dim=-1) - self.last_z1.norm(dim=-1))
            z2_embed_norm_diff = (new_z2.norm(dim=-1) - self.last_z2.norm(dim=-1))
            
            norm_diff = torch.stack([z1_embed_norm_diff, z2_embed_norm_diff]).detach().cpu()
            self.norm_change_log_temp.append(norm_diff)
            self.norm_change_log_batch.append(norm_diff.mean().item())
            
            moved_distance = ((new_z1 - self.last_z1).norm(dim=-1) + (new_z2 - self.last_z2).norm(dim=-1))
            moved_dist_mean = moved_distance.mean().detach().cpu().item()
            self.moved_dist_log_batch.append(moved_dist_mean)

            self.log("Embedding Norm Change - Batch", norm_diff.mean())

    
    def on_train_epoch_end(self):
        self.model.eval()

        cur_epoch = self.current_epoch + 1

        self.combine_epoch_embed_logs()
        mean = self.train_norm_log[-1].mean().item()
        std = self.train_norm_log[-1].std().item()
        self.log("Mean embedding norm", mean)
        self.log("Std: embedding norm", std)
        self.log("Std / Mean", (std)/(mean))

        mean_change = self.norm_change_log[-1].sum().item()
        self.log("Embedding norm change", mean_change)

        if cur_epoch % self.log_acc_every == 0 or cur_epoch == self.n_epochs:
            train_embeds = self.embed_dataset(self.knn_train_dataset)
            test_embeds = self.embed_dataset(self.knn_test_dataset)
            knn_acc = self.knn_acc(train_embeds, test_embeds)
            self.log("kNN accuracy (cosine)", knn_acc)
            self.save_logs()
                
        self.model.train()
                
    def combine_epoch_embed_logs(self):
        if len(self.idx_log_temp) > 1:
            sort_idx = torch.cat(self.idx_log_temp).argsort()
            self.norm_change_log.append(torch.cat(self.norm_change_log_temp, dim=-1)[:,sort_idx])
            self.train_norm_log.append(torch.cat(self.train_norm_log_temp, dim=-1)[:,sort_idx])
            self.grad_norm_log.append(torch.cat(self.grad_norm_log_temp, dim=-1)[:,sort_idx])
            self.idx_log_temp = []
            self.train_norm_log_temp = []
            self.grad_norm_log_temp = []
            self.norm_change_log_temp = []
    
    def save_logs(self):
        with open(f"{self.logger.log_dir}/embed_history.npy", "wb") as file:
                    np.save(
                        file,
                        dict(
                            train_norm_history = np.stack(self.train_norm_log),
                            grad_norm_history = np.stack(self.grad_norm_log),
                            nomrm_change_history = np.stack(self.norm_change_log),
                            norm_change_batch_history = np.array(self.norm_change_log_batch),
                            norm_batch_history = np.array(self.norm_log_batch),
                            moved_dist_batch_history = np.array(self.moved_dist_log_batch),
                        )
                    )


###################### TRAINING LOOP #########################

log_dir = "logs"
checkpoint_every = 100

def train_variants(
        variants,
        experiment_set_name,
        dataset_name
        ):
    """
    Trains multiple SimCLR models from zero.
    The hyperparameters have to provided in the form of a list of dictionaries.
    Each dictionary contains the hyperparameters for one model.
    To train the SimCLR with the default parameters pass [dict()] fro variants.

    variants: list of dictionaries of hyperparameters  

    experiment_set_name: name prefix used for logging  

    dataset_name: the name of the dataset that is being trained on, the dataset options are (cifar10, cifar10_unbalanced, cifar100, cifar100_unbalanced, flowers)
    """

    for variant in variants:
        
        # load parameters not used in SimCLR class
        variant_norm_function_str = variant.get("norm_function_str", "l2")
        variant_name = variant.pop("name", variant_norm_function_str)
        variant_dataset = variant.pop("dataset_name", dataset_name)
        variant_knn_dataset = variant.pop("knn_dataset", variant_dataset)


        # load SimCLR model with hyperparameters provided in the variant dict
        mod = SimCLR(dataset_name=variant_dataset, **variant)
        if variant_knn_dataset: # overwrite knn datasets if specified
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


###################### HYPERPARAMETERS FOR RUNS #########################

variants = sum([[

    # Experiments on variants that replace the L2 norm normalization step with a different mapping onto the hypersphere

    # dict(name="angle", norm_function_str = "angle", seed=seed, warmup_epochs=1),
    # dict(name="angle2", norm_function_str = "angle2", seed=seed, warmup_epochs=1),
    # dict(name="stereo", norm_function_str = "stereo", seed=seed, warmup_epochs=1),
    # dict(name="mono", norm_function_str = "mono", seed=seed, warmup_epochs=1),
    # dict(name="exp_map", norm_function_str="exp_map", seed=seed, warmup_epochs=1),

    # Experiments on setting various parameters to zero to see what causes embedding norm growth.
    
    # dict(name="no_momentum", seed=seed, weight_decay=5e-4, momentum=0, n_epochs=1000),
    # dict(name="no_schedule", seed=seed, weight_decay=5e-4, use_lr_schedule=False, n_epochs=1000),
    # dict(name="no_decay", seed=seed, weight_decay=0, n_epochs=1000),

    # Experiments on dataset affecting embedding norm

    # dict(dataset_name="cifar100_unbalanced", seed=seed, n_epochs=100),
    # dict(dataset_name="cifar100", seed=seed, n_epochs=100),
    # dict(dataset_name="cifar10", seed=seed, n_epochs=100),
    # dict dataset_name="cifar10_unbalanced", seed=seed, n_epochs=100),
    # dict(name="shuffled_positve_pairs", seed=seed, n_epochs=100, shuffle_positive_pairs=True),

    # Experiments on increasing LR without adjusting WD
    
    # dict(name="lr_ablation", seed=seed, base_lr=0.06*8, n_epochs=100,),
    # dict(name="lr_ablation", seed=seed, base_lr=0.06*16, n_epochs=100,),
    # dict(name="lr_ablation", seed=seed, base_lr=0.06*32, n_epochs=100,),
    # dict(name="lr_ablation", seed=seed, base_lr=0.06*128, n_epochs=100,),

    # Experiments on changing WD without adjusting LR

    # dict(seed=seed, weight_decay=5e-4*64, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4*4, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4*2, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4/2, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4/4, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4/8, n_epochs=100, adjust_head=True),
    # dict(seed=seed, weight_decay=5e-4/16, n_epochs=100, adjust_head=True),

    # Experiments on various LR, WD pairs with on LR schedule. This shows how the embedding norm converge to a constant value for each pair.

    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4*16, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4*8, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4*4, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4*2, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4/2, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06, weight_decay=5e-4/4, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 2, weight_decay=5e-4*16, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 2, weight_decay=5e-4*8, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 4, weight_decay=5e-4*4, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 4, weight_decay=5e-4*2, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 4, weight_decay=5e-4, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 4, weight_decay=5e-4/2, n_epochs=100, use_lr_schedule=False),
    # dict(seed=seed, base_lr=0.06 * 4, weight_decay=5e-4/4, n_epochs=100, use_lr_schedule=False),

    # Experiments on increasing LR and decreaseing WD equally

    # dict(seed=seed, base_lr=0.06*2, weight_decay=5e-4/2, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*4, weight_decay=5e-4/4, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*8, weight_decay=5e-4/8, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*32, weight_decay=5e-4/32, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*64, weight_decay=5e-4/64, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*128, weight_decay=5e-4/128, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*256, weight_decay=5e-4/256, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*1024, weight_decay=5e-4/1024, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*2048, weight_decay=5e-4/2048, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*4096, weight_decay=5e-4/4096, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*8192, weight_decay=5e-4/8192, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*16384, weight_decay=5e-4/16384, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06*(2**15), weight_decay=5e-4/(2**15), n_epochs=100, adjust_head=True),

    # Experiments on decreasing LR and increasing WD
    
    # dict(seed=seed, base_lr=0.06/2, weight_decay=5e-4*2, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/4, weight_decay=5e-4*4, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/8, weight_decay=5e-4*8, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/16, weight_decay=5e-4*16, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/32, weight_decay=5e-4*32, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/128, weight_decay=5e-4*128, n_epochs=100, adjust_head=True),
    # dict(seed=seed, base_lr=0.06/256, weight_decay=5e-4*256, n_epochs=100, adjust_head=True),

    
    # Experiment, training SimCLR with 256xLR and WD/256 for 1000 epochs.
    # Knn accuracy evovles the same as for default SimCLR.
    
    dict(seed=seed, base_lr=0.06*256, weight_decay=5e-4/256, n_epochs=1000, adjust_head=True),
    dict(seed=seed) # This trains default SimCLR for 1000 epochs with random seed=seed

] for seed in range(1,2)], [])

if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`*")

    experiment_set_name = "experiment_1"
    dataset_name = "cifar10"
    train_variants(variants, experiment_set_name, dataset_name)
    
# CUDA_VISIBLE_DEVICES=2 python lightning_simclr.py