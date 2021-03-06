#### SPATIAL TRANSFORMERS

The Spatial Transformer is a learnable module which explicitly allows the spatial manipulation of data within the network. This differentiable module can be inserted into existing convolutional architectures, giving neural networks the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process.

![image](https://user-images.githubusercontent.com/67177106/147401679-62b0bf2b-80b9-4297-aee5-b039b9082109.png)
Figure 1

The action of the spatial transformer is conditioned on individual data samples, with the appropriate behaviour learnt during training for the task in question (without extra supervision). Unlike pooling layers, where the receptive fields are fixed and local, the spatial transformer module is a dynamic mechanism that can actively spatially transform an image (or a feature map) by producing an appropriate transformation for each input sample. The transformation is then performed on the entire feature map (non-locally) and
can include scaling, cropping, rotations, as well as non-rigid deformations. This allows networks which include spatial transformers to not only select regions of an image that are most relevant (attention), but also to transform those regions to a canonical, expected pose to simplify recognition in the following layers. Notably, spatial transformers can be trained with standard back-propagation, allowing for end-to-end training of the models they are injected in.

The spatial transformer mechanism is split into three parts, shown in Figure 1 . In order of computation, 

1) First a localisation network takes the input feature map, and through a number of hidden layers outputs the parameters of the spatial transformation that should be applied to the feature map – this gives a transformation conditional on the input. The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.

2) Then, the predicted transformation parameters are used to create a sampling grid, which is a set of points where the input map should be sampled to produce the transformed output. This is done by the grid generator. The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.

3) Finally, the feature map and the sampling grid are taken as inputs to the sampler, producing the output map sampled from the input at the grid points. The sampler uses the parameters of the transformation and applies it to the input image.

The combination of the localisation network, grid generator, and sampler form a spatial transformer. This is a self-contained module which can be dropped into a CNN architecture at any point, and in any number, giving rise to spatial transformer networks. This module is computationally very fast and does not impair the training speed, causing very little time overhead when used naively, and even speedups in attentive models due to subsequent downsampling that can be applied to the output of the transformer. Placing spatial transformers within a CNN allows the network to learn how to actively transform the feature maps to help minimise the overall cost function of the network during training. The knowledge of how to transform each training sample is compressed and cached in the weights of the localisation network (and also the weights of the layers previous to a spatial transformer) during training. For some tasks, it may also be useful to feed the output of the localisation network, θ, forward to the rest of the network, as it explicitly encodes the transformation, and hence the pose, of a region or object. It is also possible to use spatial transformers to downsample or oversample a feature map, as one can define the output dimensions H0 and W0 to be different to the input dimensions H and W. However, with sampling kernels with a fixed, small spatial support (such as the bilinear kernel), downsampling with a spatial transformer can cause aliasing effects.

Finally, it is possible to have multiple spatial transformers in a CNN. Placing multiple spatial transformers at increasing depths of a network allow transformations of increasingly abstract representations, and also gives the localisation networks potentially more informative representations to base the predicted transformation parameters on. One can also use multiple spatial transformers in parallel – this can be useful if there are multiple objects or parts of interest in a feature map that should be
focussed on individually. A limitation of this architecture in a purely feed-forward network is that the number of parallel spatial transformers limits the number of objects that the network can model.

Spatial transformers can be incorporated into CNNs to benefit multifarious tasks, for example:
1) Image Classification: suppose a CNN is trained to perform multi-way classification of images according to whether they contain a particular digit – where the position and size of the digit may vary significantly with each sample (and are uncorrelated with the class); a spatial transformer that crops out and scale-normalizes the appropriate region can simplify the subsequent classification task, and lead to superior classification performance.
2) Co-Localisation: Given a set of images containing different instances of the same (but unknown) class, a spatial transformer can be used to localise them in each image.
3) Spatial Attention: A spatial transformer can be used for tasks requiring an attention mechanism but is more flexible and can be trained purely with backpropagation without reinforcement learning. A key benefit of using attention is that transformed (and so attended), lower resolution inputs can be used in favour of higher resolution raw inputs, resulting in increased computational efficiency.

Credits and References : 
1) https://arxiv.org/pdf/1506.02025.pdf (Spatial Transformer Networks)
2) https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html


Code Links 
--------------------------------------

Colab Link : https://colab.research.google.com/drive/1G88k46k4DPNeMGGsEQQGWeWol_xv5ZAf?usp=sharing

Github Link : https://github.com/sherry-ml/EVA7/blob/main/S12/Session%2012%20Assignment.ipynb

Training Logs for last few epochs
---------------------------------------
![image](https://user-images.githubusercontent.com/67177106/147410634-1cdf4b0b-dca3-4cc6-9eed-f6708c32c292.png)


Model Summary 
-------------------------------------
![image](https://user-images.githubusercontent.com/67177106/147410518-bd4a6fcf-084a-4056-bb50-5e1fe1b6d9cc.png)

###VISUALIZING THE STN RESULTS
----------------------------------------------

![image](https://user-images.githubusercontent.com/67177106/147410567-c8bea9f5-1c46-4fbe-9a0a-cddf6c1ae0db.png)

Contributors
-------------------------
Lavanya Nemani

Shaheer Fardan

