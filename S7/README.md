### SESSION 7 Assignment

Code for this assignment is divided into 6 files out of which ones is main notebook where all the code is actually executed and remaining 5 contain class definitions and other function definitions which are executed in the main notebook.

We will first go through 5 code files which contain class and function definitions:

1) model_class.py ( https://github.com/sherry-ml/EVA7/blob/main/S7/model_class.py ) : This file contains model definition for our CNN. It consists of 4 convolution blocks, 3 transition blocks and 1 output block arranged in following order:
  - Convolution Block 1
  - Transition Block 1
  - Convolution Block 2 ( First part of this block uses Depthwise Seperable Convolution)
  - Transition Block 2
  - Convolution Block 3
  - Transition Block 3 (First part of this block uses Dilated Convolution
  - Convolution Block 4
  - Output Layer ( Consists of GAP layer)
  
  Class definition takes two parameters - dropout and type of normalization to be used. We used Batch Normalization only for this assignment.
 
 2) utility.py ( https://github.com/sherry-ml/EVA7/blob/main/S7/utility.py ): This file contains definition for following functions and classes :
  - default_DL function : This function creates train and test dataset/dataloader without applying any image augmentation and returns dataloader and dataset objects for training dataset
  -  C_10_DS Class : This is class definition for loading CIFAR 10 dataset and apply custom augmentation/albumentation
  -  set_compose_params function: This function takes two parameters - mean and standard deviation. We calculate mean and standard deviation using default train dataset object returned by default_DL function. This function contains various albumentation/transformation that are to be applied to CIFAR10 dataset. It returns training and validation(test) transformation objects.
  -  tl_ts_mod function: This function takes  training and validation transformation objects as parameters(returned by set_compose_params function above). This function creates training and test datasets and dataloaders using C_10_DS class definition after applying appropriate transformations using training and validation transformation objects(passed on to this function) and returns corresponding train dataset, train dataloader, test dataset and test dataloader.

3) train_model.py (https://github.com/sherry-ml/EVA7/blob/main/S7/train_model.py): This file contains train function definition which takes following parameters : 
    - model: This is the model object corresponding to CNN that we are training
    - device: cpu or cuda
    - train_loader: This the train loader returned by tl_ts_mod function described above
    - optimizer: This is optimizer object (We are using Adam as optimizer)
    - epoch: Epoch number
    - train_losses: This is the list object passed to the function to store train losses for different epochs
    - train_acc: This is the list object passed to this function to store training accuracy for different aprochs
    - lambda_l1 : This is the value of L1 regularization constant to be applied if using L1 regularization. We are not using L1 regularization in this assignment. 
   This function iterates through train dataloader loading data in batches, passes it through model, then calculate loss and perform gradient calculation and weight adjustment. Loss and accuracy for each batch is added up till the end of epoch which is then stored inside corresponding list objects passed onto this function.
   
4) test_model.py (https://github.com/sherry-ml/EVA7/blob/main/S7/test_model.py) : This file contains test function definition which takes the following parameters:
    - model : This is the model object corresponding to CNN that we are training.
    - device : cpu or cuda
    - test_loader: This is the test loader object returned by tl_ts_mod function above
    - test_losses: List object to store test loss for different epochs
    - test_acc: List object to store test acuracies for different epochs
    - epoch: Epoch number
This function is called after end of each training epoch completion. This function passes data from test loader to trained model and then calculates loss and accuracy per batch. Test Losses and accuracies are added for all batches and then stored in corresponding list objects once all the data in test loader has been processed through model. At the end this function evaluates if the test accuracy for this epoch is greater than 85%. If yes, then it saves model weights in a file and prints its filename. This function returns test accuracy for this epoch to the calling code.

5) model_training.py (https://github.com/sherry-ml/EVA7/blob/main/S7/model_training.py) : This file contains definition for train_test_model function which takes the following parameters:
    - trainloader: This is the train dataloader that is returned by tl_ts_mod function above
    - testloader : This is the test dataloader that is returned by tl_ts_mod function above
    - norm_type='BN': Normalization type to be applied during while creating model. BN - Batch Normalization; LN - Layer Normalization; GN- Group Normalization
    - EPOCHS=20: Number of epochs for which model training has to be done. Default value is 20
    - dropout=0.1: Amount of dropout to be applied when creating CNN model Default value is 0.1
    - lr=0.001: This is the learning rate that need to specified when creating optimizer object. Default is 0.001
    - device='cpu' : Device is cpu or cuda. Default value is cpu
This function forms the main code block where each piece of participating code block mentioned above is called and integrated together. This function creates model object, optimizer objects, initializes various list object to hold train/test losses and accuracies. After creating necessary objects, it calls train function defined above passing relevant arguments followed by test function in a for loop that iterates for number of epochs passed to this function. At the end of each test function invocation, it evaluates if the test accuracy returned is greater than 85%. If yes then it breaks the loop, stores miclassified images along with their predicted and actual labels in a list object and returns model object to calling code.

Main Notebook (https://github.com/sherry-ml/EVA7/blob/main/S7/Final_Submission_Session7_Assignment.ipynb): This notebook performs following tasks in order specified below:
1) Imports and installs(if required) necessary libraries, classes and functions including custom classes and functions.
2) Calls default_DL() function and then alculates mean and std deviation of default CIFAR10 dataset
3) Calls set_compose_params(mean, std) function passing mean and std deviation calculated in step 2 above.
4) Calls tl_ts_mod(transform_train,transform_valid) function passing train and validation transform parameters obtained in step 3 above and gets trainset, , trainloader, testset, testloader objects in return.
5) Executes view_model_summary function to print model parameters summary and model definition.
6) Executes train_test_model(trainloader_mod, testloader_mod, 'BN', 200, 0.005,0.002, device ) function with values as specified. This function will exit as soon as reported test accuracy becomes greater than 85% and returns model object.
7) Model object from Step 6 above is used to calculate %age of correctly classified images belonging to each class.

--------------------------------------------------------------------------------------------------------------------------------

### Model used in this assignment:

Net(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
    (4): Conv2d(32, 64, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout2d(p=0.005, inplace=False)
  )
  (trans1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout2d(p=0.005, inplace=False)
  )
  (trans2): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
  )
  (conv3): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout2d(p=0.005, inplace=False)
  )
  (trans3): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
  )
  (conv4): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout2d(p=0.005, inplace=False)
    (4): Conv2d(16, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout2d(p=0.005, inplace=False)
  )
  (out): Sequential(
    (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
)

---------------------------------------------------------------------------------------------------------------------------
####Torch Summary

        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
         Dropout2d-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 31, 31]          32,768
       BatchNorm2d-6           [-1, 64, 31, 31]             128
              ReLU-7           [-1, 64, 31, 31]               0
         Dropout2d-8           [-1, 64, 31, 31]               0
            Conv2d-9           [-1, 32, 15, 15]          18,464
      BatchNorm2d-10           [-1, 32, 15, 15]              64
             ReLU-11           [-1, 32, 15, 15]               0
        Dropout2d-12           [-1, 32, 15, 15]               0
           Conv2d-13           [-1, 32, 15, 15]           9,216
      BatchNorm2d-14           [-1, 32, 15, 15]              64
             ReLU-15           [-1, 32, 15, 15]               0
        Dropout2d-16           [-1, 32, 15, 15]               0
           Conv2d-17           [-1, 32, 15, 15]             288
      BatchNorm2d-18           [-1, 32, 15, 15]              64
             ReLU-19           [-1, 32, 15, 15]               0
        Dropout2d-20           [-1, 32, 15, 15]               0
           Conv2d-21             [-1, 16, 7, 7]           4,624
      BatchNorm2d-22             [-1, 16, 7, 7]              32
             ReLU-23             [-1, 16, 7, 7]               0
        Dropout2d-24             [-1, 16, 7, 7]               0
           Conv2d-25             [-1, 32, 5, 5]           4,608
      BatchNorm2d-26             [-1, 32, 5, 5]              64
             ReLU-27             [-1, 32, 5, 5]               0
        Dropout2d-28             [-1, 32, 5, 5]               0
           Conv2d-29             [-1, 32, 5, 5]           9,216
      BatchNorm2d-30             [-1, 32, 5, 5]              64
             ReLU-31             [-1, 32, 5, 5]               0
        Dropout2d-32             [-1, 32, 5, 5]               0
           Conv2d-33             [-1, 16, 2, 2]           4,624
      BatchNorm2d-34             [-1, 16, 2, 2]              32
             ReLU-35             [-1, 16, 2, 2]               0
        Dropout2d-36             [-1, 16, 2, 2]               0
           Conv2d-37             [-1, 16, 2, 2]           2,304
      BatchNorm2d-38             [-1, 16, 2, 2]              32
             ReLU-39             [-1, 16, 2, 2]               0
        Dropout2d-40             [-1, 16, 2, 2]               0
           Conv2d-41             [-1, 10, 2, 2]           1,440
      BatchNorm2d-42             [-1, 10, 2, 2]              20
             ReLU-43             [-1, 10, 2, 2]               0
        Dropout2d-44             [-1, 10, 2, 2]               0
        AvgPool2d-45             [-1, 10, 1, 1]               0
================================================================
Total params: 89,044
Trainable params: 89,044
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.61
Params size (MB): 0.34
Estimated Total Size (MB): 3.97
----------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------

### Training Logs for last few epochs:


EPOCH: 134
100%|██████████| 391/391 [00:11<00:00, 33.01it/s]
 Average Training Loss=0.753562908782959, Accuracy=80.672
Test set: Average loss: 0.4584, Accuracy: 8472/10000 (84.72%)

EPOCH: 135
100%|██████████| 391/391 [00:11<00:00, 33.64it/s]
 Average Training Loss=0.756516551361084, Accuracy=80.652
Test set: Average loss: 0.4917, Accuracy: 8349/10000 (83.49%)

EPOCH: 136
100%|██████████| 391/391 [00:11<00:00, 32.99it/s]
 Average Training Loss=0.7552795533752441, Accuracy=80.596
Test set: Average loss: 0.4706, Accuracy: 8403/10000 (84.03%)

EPOCH: 137
100%|██████████| 391/391 [00:11<00:00, 32.62it/s]
 Average Training Loss=0.7552378245544433, Accuracy=80.63
Test set: Average loss: 0.4507, Accuracy: 8483/10000 (84.83%)

EPOCH: 138
100%|██████████| 391/391 [00:11<00:00, 33.14it/s]
 Average Training Loss=0.7606863624572754, Accuracy=80.468
Test set: Average loss: 0.4525, Accuracy: 8445/10000 (84.45%)

EPOCH: 139
100%|██████████| 391/391 [00:11<00:00, 33.13it/s]
 Average Training Loss=0.7505462648773193, Accuracy=80.798
Test set: Average loss: 0.4453, Accuracy: 8517/10000 (85.17%)

Saved Model weights in file:  Session7_assignment_epoch_139_acc_85.17.pth
Total Number of incorrectly predicted images by model type BN is 1483

#################################################################################################################################################################

%age of images classified correctly per class
Accuracy of plane : 82 %
Accuracy of   car : 92 %
Accuracy of  bird : 75 %
Accuracy of   cat : 67 %
Accuracy of  deer : 77 %
Accuracy of   dog : 81 %
Accuracy of  frog : 88 %
Accuracy of horse : 92 %
Accuracy of  ship : 93 %
Accuracy of truck : 87 %
