### SESSION 7 Assignment

Code for this assignment is divided into 6 files out of which ones is main notebook where all the code is actually executed and remaining 5 contain class definitions and other function definitions which are executed in the main notebook.

We will first go through 5 code files which contain class and function definitions:

1) model_class.py ( https://github.com/sherry-ml/EVA7/blob/main/S7/model_class.py ) : This file contains model definition for our CNN. It consists of 4 convolution blocks, 3 transition blocks and 1 output block arranged in following order:
  - Convolution Block 1
  - Transition Block 1
  - Convolution Block 2 ( Second part of this block uses Depthwise Separable Convolution)
  - Transition Block 2
  - Convolution Block 3 (First part of this block uses Dilated Convolution)
  - Transition Block 3 
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


This function forms the main code block where each piece of participating code block mentioned above is called and integrated together. This function creates model object, optimizer objects, initializes various list object to hold train/test losses and accuracies. After creating necessary objects, it calls train function defined above passing relevant arguments followed by test function in a for loop that iterates for number of epochs passed to this function. At the end of each test function invocation, it evaluates if the test accuracy returned is greater than 85%. If yes then it breaks the loop, stores misclassified images along with their predicted and actual labels in a list object and returns model object to calling code.

Main Notebook (https://github.com/sherry-ml/EVA7/blob/main/S7/Final_Submission_Session7_Assignment.ipynb): This notebook performs following tasks in order specified below:
1) Imports and installs(if required) necessary libraries, classes and functions including custom classes and functions.
2) Calls default_DL() function and then calculates mean and std deviation of default CIFAR10 dataset
3) Calls set_compose_params(mean, std) function passing mean and std deviation calculated in step 2 above.
4) Calls tl_ts_mod(transform_train,transform_valid) function passing train and validation transform parameters obtained in step 3 above and gets trainset, , trainloader, testset, testloader objects in return.
5) Executes view_model_summary function to print model parameters summary and model definition.
6) Executes train_test_model(trainloader_mod, testloader_mod, 'BN', 200, 0.005,0.002, device ) function with values as specified. This function will exit as soon as reported test accuracy becomes greater than 85% and returns model object.
7) Model object from Step 6 above is used to calculate %age of correctly classified images belonging to each class.

--------------------------------------------------------------------------------------------------------------------------------

### Model used in this assignment:
![image](https://user-images.githubusercontent.com/67177106/141820023-e854ea4a-b575-437c-b004-0e83dbe51444.png)
---------------------------------------------------------------------------------------------------------------------------
####Torch Summary

![image](https://user-images.githubusercontent.com/67177106/141820388-b208f864-361e-429b-be55-a80e712fc84c.png)
-----------------------------------------------------------------------------------------------------------------------

### Training Logs for last few epochs:

![image](https://user-images.githubusercontent.com/67177106/141820621-b62462f9-e6dd-4dc4-a930-6d2e08fe7570.png)
#################################################################################################################################################################

%age of images classified correctly per class

![image](https://user-images.githubusercontent.com/67177106/141820710-e53e575b-cd66-4705-8841-e3632780dadd.png)
