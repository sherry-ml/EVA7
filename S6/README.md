# SESSION 6 Assignment

### PART 1
Code is divided into two files- model_class.py (https://github.com/sherry-ml/EVA7/blob/main/S6/model_class.py) and Normalization_Session_6_Assignment.ipynb (https://github.com/sherry-ml/EVA7/blob/main/S6/Normalization_Session_6_Assignment.ipynb)

model_class.py contains class with model specifications. Class takes argument that will indicate what type of normalization you want to perform and accordingly it will apply the corresponding normalization technique. This class is imported inside the Normalization_Session_6_Assignment.ipynb.
  - BN for Batch Normalization
  - LN for Layer Normalization (All channels in single image in single group)
  - GN for Group Normalization  ( Channels in single image are divided into two groups)

Normalization_Session_6_Assignment.ipynb contains main body of the code. It imports necessary libraries, creates pytorch datasets and dataloaders.
  - It contains function view_model_summary() which takes normalization type as one of the arguments and displays summary of model parameters and layers.
  - It contains generic train() and test() functions which can be called passsing specific values of model object and other parameters.
  - It contains train_test_model() function which takes normalization type as one of the arguments. Default is Batch Normmalization. This function serves the main code body which initializes several parameters, instantiates model and calls train and test function with relevant parameters. Once the training is completed, it calculates number of misclassified images particular to normalization technique and stores missclassified image, its predicted label and actual label as tuple in a list object. train_test_model function is then called with different normalization technique values to train model with corresponding normalization technique. We store train and test losses and accuracies in relevent list objects.
  - It contains display_incorrect_images function that takes list containing misclassified images and then displays first 10 misclassified images in 5X2 matrix.
  - Last, it displays relevant graphs for training/test losses and accuracies for models using different normalization techniques
 
## PART 2
Different Normalization techniques applied in Excel sheets. Snapshots shown below:

Batch Normalization
![image](https://user-images.githubusercontent.com/67177106/139722902-0229b476-0ae3-465c-9614-7d2bfc7f185d.png)

Layer Normalization
![image](https://user-images.githubusercontent.com/67177106/139723058-ae87e659-0406-44e0-a214-2896924c284c.png)

Group Normalization
![image](https://user-images.githubusercontent.com/67177106/139723180-60f83d32-268f-40dc-ac08-799c35abe8cb.png)

### Part 3 
Findings for normalization techniques
To be Added

### Part 4

Graphs showing Training/Test Loss/Accuracies

![image](https://user-images.githubusercontent.com/67177106/139723387-93110a4c-8fbe-42cf-82a0-6e882219df5d.png)

### Part 5
 Collection of misclassified images
 
 Batch Normalization misclassified images
 
 ![image](https://user-images.githubusercontent.com/67177106/139723533-86725493-1d90-4339-b961-b1701d4fde15.png)

Layer Normalization misclassified images

![image](https://user-images.githubusercontent.com/67177106/139723654-49385af8-719e-4eac-8ba9-7d4920b10c5c.png)

Group Normalization misclassified images

![image](https://user-images.githubusercontent.com/67177106/139723740-dbc44346-4f13-4c2c-b8a5-74ccb68e8f3e.png)

