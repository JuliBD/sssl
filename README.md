# Spherical Selfsupervised Learning  
(repository for my Master Thesis)  

My Master Thesis started with the question "Can we improve the performance of SimCLR, by replacing the normalization with a direct mapping onto the hypersphere?". In SimCLR the embedding representation of images are normalized (mapped onto the hypersphere) as a part of calculating the cosine-similarity metric which is used in the loss. However, the normalization removes one degree of freedom. This means after normalization the norm dimension is irrelevant to the loss, meaning the embedding vectors $v$ and $2 \cdot v$ will lead to the same loss. Even thought the loss value does not change depending on the norm of the embeddings, the gradient does very much change dependent on the norm of the embeddings (see: Andrews paper TODO: add link).

This repository contains three sets of experiments on:  
- Improving performance of SimCLR by replacing normalization witha direct mapping on to the hypersphere  
- What parts of SimCLR training influence the embedding norm dynamics during training  
- Investigating how much the embedding norm depends on the model vs the data (images) 