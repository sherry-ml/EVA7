# Session 13 Assignment

Vision Transformer 
-------------------
Paper : https://arxiv.org/pdf/2010.11929.pdf

![image](https://user-images.githubusercontent.com/67177106/149081898-fcc430e2-dc27-4d20-a74f-68af930670e8.png)


Brief description of the code
------------------------------
1) Code starts by downloading, unzipping train, test data and splitting them into train, validation and test data.
3) Class CatsDogsDataset takes data as argument and performs required transformations as applicable in case of train, validation and test data and returns data in form of dataset object which is further used to create train, validation and test data loader objects.
4) Thereafter, Vision transformer object is created. It divides the source 224x224 image into chunks of 32x32 which effectively translates into 49 patches. Add 1 more patch to account for classification, it makes total of 50 patches. 
5) VIT class integrates patch embedding with 12 transformer blocks created using Linformer object and then adds a linear layer on top of 12 transformer blocks which is used for final image classification.
6) Each transformer block consists of two parts : a) three linear layers accounting for query, key and value blocks followed by dropout layer (0.0 probability so effectively no dropout) which is followed by a linear layer and then layer normalization. b) Feed Forward neural network using two Linear layers with 0 dropout and GeLu activation function which is followed by layer normalization.
7) Output from final transformer block is fed into linear layer which is used to output logits for  2 different classes. 
8) It is not very clear from the code provided in blog on how the data flows from one transformer block to another or even within a single transformer block but it is assumed that it follows the flow as described in ViT research paper on which the blog is based.
9) Loss function used is CrossEntropyLoss, optimizer is Adam in combination StepLR scheduler.
10) Training process used in normal one. Take the input for dataloader, move the data and label to gpu, input the data into model and feed the output data along with original data into loss function following by zeroing out gradients, updating gradients and then updating weights and bias values. This followed by  calculating accuracy and loss for the epoch.


Training Logs
---------------
![image](https://user-images.githubusercontent.com/67177106/149080757-9f6dc298-f293-4780-9f51-d0532adae26e.png)


References:

1) https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/
2) ViT Paper: https://arxiv.org/pdf/2010.11929.pdf
