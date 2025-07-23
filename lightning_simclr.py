import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from norm_functions import *
from data_utils import *


###################### PARAMS ##############################

NORM_FUNCTION_DICT = {
                        'l2': F.normalize, # if no norm_function_str is provided this will be used (default SimCLR)
                        'angle': angle_normalize,
                        "angle2": angle2_normalize,
                        'exp_map': exp_map_normalize,
                        "minus_exp_map": minus_exp_map_normalize,
                        'stereo': stereo_normalize,
                        "mono": mono_normalize,
                        "minus_mono": minus_mono_normalize,
                        "rescale": rescale_normalize,
                        "line": line_normalize,
                        "exp2": exp2_normalize,
                        "exponent": exponent_normalize,
                        "torus": torus_norm,
                    }



###################### NETWORK ARCHITECTURE #########################

class ResNet18withProjector(nn.Module):
    def __init__(
            self, 
            dataset_name, 
            projector_hidden_size=1024
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
            nn.Linear(projector_hidden_size, 128),
        )

        self.certainty_head = nn.Sequential(
            nn.Linear(512, projector_hidden_size), 
            nn.ReLU(), 
            nn.Linear(projector_hidden_size, 128),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)

        c = self.certainty_head(h)
        return h, z, c
        


# def infoNCE(
#         features,
#         datapoint_idx,
#         temperature=0.5, 
#         norm_function=F.normalize,
#         power = 0
#         ):
#     batch_size = features.size(0) // 2

#     features = RescaleNorm.apply(features, datapoint_idx, power)
#     x = norm_function(features)
#     cos_xx = x @ x.T / temperature

#     cos_xx.fill_diagonal_(float("-inf"))
        
#     targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
#     targets[:batch_size] += batch_size
#     targets[batch_size:] -= batch_size

#     return F.cross_entropy(cos_xx, targets)


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
        self.base_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)
    
    def get_lr(self):
        self.cur_epoch += 1
        cur_epoch = self.cur_epoch

        if cur_epoch < self.warmup_epochs + 1:
            # linear reascaling of lr between warmup and the cosine annealing lr value
            lr = [np.linspace(self.warmup_lr, self.base_lr, self.warmup_epochs + 1)[cur_epoch] for _ in super().get_lr()]
        else:
            lr = super().get_lr()
        return lr


##################### ANDREW'S MODIFICATIONS #########################

# class RescaleNorm(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, z, datapoint_idx, power=0):
#         ctx.save_for_backward(z)
#         ctx.save_for_backward(datapoint_idx)
#         ctx.power = power
#         return z

#     @staticmethod
#     def backward(ctx, grad_output):
#         z = ctx.saved_tensors[0]
#         datapoint_idx = ctx.saved_tensors[1]
#         power = ctx.power
#         norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)

#         return grad_output * norm**power, None

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
            momentum = 0.9,
            nestrov = False,
            n_epochs = 1000,
            norm_function_str = "l2",
            warmup_epochs = 1,
            log_acc_every = 10,
            record_embed_histories = True,
            cut = 1,
            power = 0,
            seed = get_truly_random_seed_through_os(),
            use_lr_schedule = True,
            add_certainty = False
                 ):
        super().__init__()

        ignore_list = ["log_acc_every", "record_embed_histories"]
        self.save_hyperparameters(ignore=ignore_list)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ResNet18withProjector(dataset_name)
        cut_weights(self.model, cut)

        def infoNCE(
                datapoint_idx,
                features,
                certainty,
                temperature=0.5, 
                norm_function=F.normalize,
                power=0
                ):
            batch_size = features.size(0) // 2

            features = RescaleNorm.apply(datapoint_idx, features, power)
            x = norm_function(features) * certainty
            cos_xx = x @ x.T / temperature

            cos_xx.fill_diagonal_(float("-inf"))
                
            targets = torch.arange(cos_xx.size(0), dtype=int, device=cos_xx.device)
            targets[:batch_size] += batch_size
            targets[batch_size:] -= batch_size

            return F.cross_entropy(cos_xx, targets)

        self.norm_function = NORM_FUNCTION_DICT[norm_function_str]
        self.loss = lambda idx, features, certainty: infoNCE(idx, features, certainty, norm_function=lambda norm: self.norm_function(norm), power=power)
        
        self.record_embed_histories = record_embed_histories
        self.idx_history_temp = []
        self.grad_norm_history_temp = []
        self.grad_norm_history = []
        self.train_norm_history_temp = []
        self.train_norm_history = []
        
        class RescaleNorm(torch.autograd.Function):
            @staticmethod
            def forward(ctx, datapoint_idx, z, power=0):
                ctx.save_for_backward(datapoint_idx, z)
                ctx.power = power
                return z

            @staticmethod
            def backward(ctx, grad_output):
                datapoint_idx, z = ctx.saved_tensors
                power = ctx.power
                norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
                
                if self.record_embed_histories:
                    n = int(norm.shape[0]/2)
                    view1_norm = norm[:n].flatten().detach().cpu()
                    view2_norm = norm[n:].flatten().detach().cpu()
                    self.train_norm_history_temp.append(torch.stack([view1_norm, view2_norm]))

                    self.idx_history_temp.append(datapoint_idx.detach().cpu())

                    grad_norm = torch.linalg.vector_norm(grad_output, dim=-1).detach().cpu()
                    view1_grad_norm = grad_norm[:n]
                    view2_grad_norm = grad_norm[n:]
                    self.grad_norm_history_temp.append(torch.stack([view1_grad_norm, view2_grad_norm]))

                return None, grad_output * norm**power, None
            
            # @staticmethod
            # def save_histories(datapoint_idx, norm, grad_output):
            #     # recording various histories
            #     n = int(norm.shape[0]/2)
            #     view1_norm = norm[:n].flatten().detach().cpu()
            #     view2_norm = norm[n:].flatten().detach().cpu()
            #     self.train_norm_history_temp.append(torch.stack([view1_norm, view2_norm]))

            #     self.idx_history_temp.append(datapoint_idx.detach().cpu())

            #     grad_norm = torch.linalg.vector_norm(grad_output, dim=-1).detach().cpu()
            #     view1_grad_norm = grad_norm[:n]
            #     view2_grad_norm = grad_norm[n:]
            #     self.grad_norm_history_temp.append(torch.stack([view1_grad_norm, view2_grad_norm]))
                
            #     view1_grad = grad_output[:n]
            #     view2_grad = grad_output[n:]
            #     combined_grad = view1_grad + view2_grad
            #     combined_grad_norm = torch.linalg.vector_norm(combined_grad, dim=-1).detach().cpu()
            #     self.combined_grad_norm_history_temp.append(combined_grad_norm)

        

        dataset = get_dataset(dataset_name)
        self.knn_train_dataset, self.knn_test_dataset = get_train_and_test_set(dataset)

        self.previous_embeds_on_sphere = None
        self.distances_history = []
        self.norm_history = []


        self.cut = cut
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
        self.add_certainty = add_certainty

    
    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.base_lr * self.batch_size / 256,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nestrov,
        )
        if self.use_lr_schedule:
            lr_scheduler = {
                'scheduler': CosineAnnealingWarmup(
                                optimizer,
                                T_max=self.n_epochs,
                                warmup_epochs=self.warmup_epochs
                                ),
                'name': 'Cosine_Annealing_with_Warmup'
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
    
    def training_step(self, batch):
        idx, view1, view2, _ = batch
        view1 = view1
        view2 = view2
        
        _, z1, c1 = self.model(view1)
        _, z2, c2 = self.model(view2)

        loss = self.loss(idx, torch.cat((z1, z2)), torch.cat((c1,c2)) if self.add_certainty else 1)

        if len(view1) == self.batch_size: # only log batches with full size
            self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

###################### EVALUATION #########################

    def embed_dataset(self, dataset):
        with torch.no_grad():
            X, y, Z = [], [], []

            for batch in DataLoader(dataset, batch_size=1024):
                images, labels = batch

                h, z, c = self.model(images.to(self.device))

                X.append(h.cpu().numpy())
                Z.append(z.cpu().numpy())
                y.append(labels)

            X = np.vstack(X)
            y = np.hstack(y)
            Z = np.vstack(Z)

            return X, y, Z

    def log_embed_histories(self, projector_embeddings_np):
        if not self.record_embed_histories: return None
        
        if len(self.idx_history_temp) > 1:
            sort_idx = torch.cat(self.idx_history_temp).argsort()
            self.train_norm_history.append(torch.cat(self.train_norm_history_temp, dim=-1)[:,sort_idx])
            self.grad_norm_history.append(torch.cat(self.grad_norm_history_temp, dim=-1)[:,sort_idx])
            self.idx_history_temp = []
            self.train_norm_history_temp = []
            self.grad_norm_history_temp = []

        with torch.no_grad():
            projector_embeddings = torch.tensor(projector_embeddings_np)
            embeddings_on_sphere = self.norm_function(projector_embeddings).cpu().numpy()

        if self.previous_embeds_on_sphere is None:
            self.previous_embeds_on_sphere = embeddings_on_sphere
            new_norms = np.linalg.norm(projector_embeddings_np, axis=1)
            self.norm_history.append(new_norms)
        else:
            new_distances = np.arccos(np.sum(embeddings_on_sphere * self.previous_embeds_on_sphere, axis=1))
            new_norms = np.linalg.norm(projector_embeddings_np, axis=1)
            self.norm_history.append(new_norms)
            self.distances_history.append(new_distances)
            self.previous_embeds_on_sphere = embeddings_on_sphere
    
    def save_histories(self):
        with open(f"{self.logger.log_dir}/embed_history.npy", "wb") as file:
                    np.save(
                        file,
                        dict(
                            distance_history = np.vstack(self.distances_history),
                            norm_history = np.vstack(self.norm_history),
                            train_norm_history = np.stack(self.train_norm_history),
                            grad_norm_history = np.stack(self.grad_norm_history),
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

        self.log_embed_histories(train_embeds[-1])
        knn_acc = self.knn_acc(train_embeds,test_embeds)

        self.logger.log_hyperparams(self.hparams, {"kNN accuracy (cosine)": knn_acc})
        return super().on_fit_start()
    
    def on_train_epoch_end(self):
        self.model.eval()

        cur_epoch = self.current_epoch + 1
        train_embeds = self.embed_dataset(self.knn_train_dataset)
        test_embeds = self.embed_dataset(self.knn_test_dataset)

        self.log_embed_histories(train_embeds[-1])

        if cur_epoch % self.log_acc_every == 0 or cur_epoch == self.n_epochs:
            knn_acc = self.knn_acc(train_embeds,test_embeds)
            self.log("kNN accuracy (cosine)", knn_acc)
            self.save_histories()
                
        self.model.train()


###################### ACTUAL TRAINING #########################

log_dir = "logs"
checkpoint_every = 100

def train_variants(variants, experiment_set_name, dataset_name):

    for variant in variants:
        
        variant_name = variant.pop("name", variant["norm_function_str"])
        mod = SimCLR(dataset_name=dataset_name, **variant)
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
    # dict(norm_function_str = "l2", seed=seed, weight_decay=0, momentum=0),
    dict(norm_function_str = "l2", seed=seed, n_epochs=100),
    # dict(norm_function_str = "l2", seed=seed, add_certainty=True, n_epochs=1000),
    # dict(norm_function_str = "l2", seed=seed, weight_decay=0),
    # dict(norm_function_str = "l2", seed=seed, use_lr_schedule=False),
    # dict(norm_function_str = "l2", seed=seed, use_lr_schedule=False, weight_decay=0),
] for seed in range(1,2)], [])

if __name__ == "__main__":
    experiment_set_name = "No_schedule-_No_decay-1000"
    dataset_name = "cifar10"
    train_variants(variants, experiment_set_name, dataset_name)
    
# CUDA_VISIBLE_DEVICES=2 python lightning_simclr.py