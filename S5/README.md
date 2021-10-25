# Session 5 Assignment

#################################################################################################

### Step 1 : Reduce number of parameters in model.
__________________________________________________

Link : https://github.com/sherry-ml/EVA7/blob/main/S5/Step%201%20Reduced%20Parameters%20Session%205%20Assignment.ipynb

1) Target:
Number of parameters in basic setup is extremely high. We need to reduce this.

2) Results:

Number of Parameters: 9380

Best Train Accuracy:98.59

Best Test Accuracy:98.67

3) Analysis:

Model is lighter than basic setup.

Both Test and Train accuracy has gone down. But this is expected as we have reduced number of parameters drastically.

Next we will further try to reduce number of parameters and see if we can maintain the same level of train and test accuracy

_________________________________________________________________________________________________________________________________________________________________________

### Step 1.1 : Bring down the number of parameters further.
___________________________________________________________

Link : https://github.com/sherry-ml/EVA7/blob/main/S5/Step%201.1%20Further%20Reduce%20Parameters%20Session%205%20Assignment.ipynb

1) Target:

We further want to reduce the number of parameters.

2) Results:

Number of Parameters: 5,544 (previous 9380)

Best Train Accuracy: 98.34 (previous 98.59)

Best Test Accuracy:98.46 (previous 98.67)

3) Analysis:

We have further reduced the number of parameters by approx 41%

Both Test and Train accuracy has gone down slightly. Considering the number of parameters we got rid of, this minimal drop doesnt seems to be of much concern at this point of time.

Next we will introduce Batch Normalization and see if it improves the performance.

###########################################################################################

### Step 2: Increase model performance by adding Batch Normalization
____________________________________________________________________

Link : https://github.com/sherry-ml/EVA7/blob/main/S5/Step%202%20Batch%20Normalization%20Session%205%20Assignment.ipynb

1) Target:

We want increase the performance of the model. To accomplish this, we will introduce Batch Normalization

2) Results:

Number of Parameters: 5,688 (previous5,544)

Best Train Accuracy: 99.22 (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.13 (previous98.46) (previous 98.56)

3) Analysis:

There is slight increase increase in number of parameters. This is because of the introduction of Batch Normalization

Both Test and Train accuracy increased.

Model is overfitting.

There seems to be some difference between train and test accuracy. We want to further regularize the training process. We will introduce dropout next.

###############################################################################################

### Step 3: Regularize model training by adding dropouts.
_________________________________________________________

Link: https://github.com/sherry-ml/EVA7/blob/main/S5/Step%203%20Introduction%20of%20Dropouts%20Session%205%20Assignment.ipynb

1) Target:

We want to regularize the training process further. To accomplish this, we will introduce dropouts.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy: 98.36 (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.12 (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

There is a drop in training accuracy. This is expected as training parameters become more generalized as we regularize the training process.

Very minimal drop in test accuracy(0.01)

Model is underfitting now.

Next we will reduce dropout probability to see if it increases the performance

___________________________________________________________________________________________________________________________________________________________________

### Step 3.1 : Change Dropout probability to increase model performance.
________________________________________________________________________

Link : https://github.com/sherry-ml/EVA7/blob/main/S5/Step%203.1%20Reduce%20Dropout%20Probability%20Session%205%20Assignment.ipynb

1) Target:

Training accuracy dropped down in previous step. We want training accuracy to increase further. To accomplish this we will lower down dropout probability.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy: 98.99 (was 98.36) (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.2 (was 99.12) (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

We played around with dropout probabilities and lowered it down to just 2%.

There is an increase in training accuracy.

There is an increase in test accuracy.

Model is still underfitting but to a lesser extent than it was in previous step.

Increase in training and test performance and reduction of extent of underfitting points to the importance of tuning dropouts properly.

However, we are still below our target accuracy of 99.4 in 15 epochs. We will try to introduce some image augmentation in next step.

########################################################################################

Step 4 : Add Image augmentation.
________________________________

Link: https://github.com/sherry-ml/EVA7/blob/main/S5/Step%204%20Image%20Augmentation%20Session%205%20Assignment.ipynb

1) Target:

We want increase the training and test performance. To accomplish this goal, we introduce random rotation.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy: 98.71 (was 98.99) (was 98.36) (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.17 (was 99.2) (was 99.12) (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

We introduced random rotation.

There is a slight decrease in training accuracy.

There is very minimal drop in test accuracy.

Image augmentation has not helped here.

We are still below our target accuracy of 99.4 in 15 epochs. We will try to play around with learning rates and see if it helps in increasing the performance.

#####################################################################################################

Step 5: Increase training and test performance by increasing learning rate.
____________________________________________________________________________

Link: https://github.com/sherry-ml/EVA7/blob/main/S5/Step%205%20Change%20Learning%20Rate%20Session%205%20Assignment.ipynb

1) Target:

We want to increase the training and test performance. To accomplish this goal, we will play with learning rates here.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy:98.95 (was 98.71) (was 98.99) (was 98.36) (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.32 (was 99.17) (was 99.2) (was 99.12) (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

We changed learning rate to 0.03 from 0.01.

There is an increase in training accuracy.

There is an increase in test accuracy. We reached 99% test accuracy right in 6th epoch.

We are more close to our target for 99.4% test accuracy but still short of it. Next we will introduce mechanism to change learning rates as the one cycle of epoch goes on.

_______________________________________________________________________________________________________________________________________________________________________

Step 5.1 : Increase training and test performance by introducing variable learning rates(OneCycleLR)
______________________________________________________________________________________________________

Link: https://github.com/sherry-ml/EVA7/blob/main/S5/Step%205.1%20Introduce%20Variable%20Learning%20Rates%20Session%205%20Assignment.ipynb

1) Target:

We want to increase the training and test performance. To accomplish this goal, we will introduce variable learning rates here.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy:99.19 ( was 98.95) (was 98.71) (was 98.99) (was 98.36) (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.43 (was 99.32) (was 99.17) (was 99.2) (was 99.12) (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

We introduced one cycle learning rate that will vary learning rate as the batches progress within an epoch

We changed learning rate to 0.05 from 0.03.

There is an increase in training accuracy.

There is an increase in test accuracy. We have hit the target of 99.4 test accuracy in 15 epochs or less.

However, we need to show 99.4% test accuracy consistently in last few epochs. We will try to change our optimizer in next step and see if that helps in accomplishing out target result.

__________________________________________________________________________________________________________________________________________________________________

Step 5.2 : Increase training and test performane by changing optimizer from SGD to Adam
_________________________________________________________________________________________

Link: https://github.com/sherry-ml/EVA7/blob/main/S5/Step%205.2%20Change%20Optimizer%20Session%205%20Assignment.ipynb

1) Target:

We have touched magic figure of 99.4. However, we are not able to repeat this quite a few times. We are going to change optimizer to Adam as it is faster than SGD and usually works good with default parameters. We will combine Adam with OneCycleLR. This should help us reach 99.4% accuracy more often within 15 epochs limit.

2) Results:

Number of Parameters: 5,688 (previous 5,544)

Best Train Accuracy:99.44 (was 99.19) ( was 98.95) (was 98.71) (was 98.99) (was 98.36) (was 99.22) (previous 98.34) (previous 98.59)

Best Test Accuracy: 99.45 (was 99.43) (was 99.32) (was 99.17) (was 99.2) (was 99.12) (was 99.13) (previous98.46) (previous 98.56)

3) Analysis:

Number of parameters remain same.

We changed optimizer to Adam and combined it with OneCycleLR.

There is an increase in training accuracy.

There is a marginal increase in test accuracy. We have hit the target of 99.4 test accuracy 4 times in 15 epochs.

Difference between train and test accuracy in last epoch is just.01 which indicates there is almost no underfitting or overfitting

#############################################################################################################################################################################
