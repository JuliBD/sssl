# Spherical Selfsupervised Learning  
(repository for my Master Thesis)  

My Master Thesis started with the question "Can we improve the performance of SimCLR, by replacing the normalization with a direct mapping onto the hypersphere?". In SimCLR the embedding representation of images are normalized (mapped onto the hypersphere) as a part of calculating the cosine-similarity metric which is used in the loss. However, the normalization removes one degree of freedom. This means after normalization the norm dimension is irrelevant to the loss, meaning the embedding vectors $v$ and $2 \cdot v$ will lead to the same loss. Even thought the loss value does not change depending on the norm of the embeddings, the gradient does very much change dependent on the norm of the embeddings (see: [On the Importance of Embedding Norms in Self-Supervised Learning](https://arxiv.org/abs/2502.09252)).

The main file for running all the experiments is lightning_simclr.py. At the end of the file the various experiments are listed, to run one set of experiments you can un-comment the respective lines. By default if you run the lightning_simclr.py file a standard SimCLR model and a model with 256xLR and WD/256 will be trained. Both models, when trained will have a very similar accuracy. 

This repository contains three sets of experiments on:  
- Improving performance of SimCLR by replacing normalization with a direct mapping on to the hypersphere  
- What parts of SimCLR training influence the embedding norm dynamics during training  
- Investigating how much the embedding norm depends on the model vs the data (images) 