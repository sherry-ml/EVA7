# Assignments Session 4
----------------------------

Assignment Part 1
------------------

![image](https://user-images.githubusercontent.com/67177106/137684719-641b7ab2-e3e5-4ddf-865c-97aef86dfaa6.png)



We have been provided with initial parameters as under

    w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3

    w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55

    i1 = 0.05, i2 =0.1

    t1 = 0.01, t2 = 099

where w{i} are weights, i{i} are inputs, t{i} are target values

## Feedforward Process
--------------------------
We have two neurons in first layers. These act as small temporary computation unit aggregating incoming input values and weights. Their calculation is shown as under:

    h1 =w1*i1+w2+i2
  
    h2 =w3*i1+w4*i2

Next, we apply activation function on top of h1 and h2. This is to add non-linearity in the computations.

    a_h1 = σ(h1) = 1/(1+exp(-h1))
  
    a_h2 = σ(h2) = 1/(1+exp(-h2))

a_h1 and a_h2 now act as input to neurons in next layer. Since we have only input and output layer. a_h1 and a_h2 act as input to neurons in output layer (o1 and o2)

    o1 = w5 * a_h1 + w6 * a_h2
  
    o2 = w7 * a_h1 + w8 * a_h2
  
  Next we apply sigmoid function on top of o1 and o2
  
    a_o1 = σ(o1) = 1/(1+exp(-o1))
  
    a_o2 = σ(o2) = 1/(1+exp(-o2))

This completed our first feedforward preocess.

## Error Computation
----------------------
From the final values obtained from two neorons in output layer, we need to compute loss value. Loss value is the difference between output value and desired output value. Now if we just take absolute value of the different it is called L1 Loss. If we square the difference, it is called L2 Loss. We will be using L2 loss. This can be computed as under:

    E1 = ½ * ( t1 - a_o1)²

    E2 = ½ * ( t2 - a_o2)²

    E_Total = E1 + E2

Here is 1/2 is included only for mathematical convenience.

## Backpropagation
--------------------
The way backpropagation works is: 
  1) First we calculate the gradient of loss function with respect to each of the weights. Here we are excluding biases since we assume those are set to False. This step is performed when we write loss.backward() in pytorch code.
  2) Next, we adjust weights such that new weight is equal to difference between old weight and product of learning rate with gradient of loss function with respect to respective weight. So, reaching optimum weight value is dependent on two things -- how steep the gradient is and how big the step is (learning rate). When learning rate is set too high, it may just oscillate from one side to another overshoooting the loss minima. Setting it too low will cause the process to reach loss minima very slow. Weights are recalculated when we perform optimizer.step() function in pytorch.

So, in this case we will have to calculate gradient of E_total with respect to all weights(w1,w2,w3,w4,w5,w6,w7,w8)

We start first with weights closest to last layer. 

    δE_total/δw5 = δ(E1 +E2)/δw5

    δE_total/δw5 = δ(E1)/δw5       # removing E2 as there is no impact from E2 wrt w5	

             = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)	# Using Chain Rule
             
             = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))   # calculate how much does the output of a_o1 change with respect Error
             
                * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                       # calculate how much does the output of o1 change with respect a_o1
                
                * (1 - a_o1 )) * a_h1                                            # calculate how much does the output of w5 change with respect o1
                
             = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1

    δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1

    δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2

    δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1

    δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2

Next we calculate gradients with respect to weights in the next layer which is the first layer and the only layer left.

    δE_total/δa_h1 = δ(E1+E2)/δa_h1 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
               
    δE_total/δa_h2 = δ(E1+E2)/δa_h2 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8

    δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
                 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
             

    δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2

    δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1

    δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2

Now that we have gradient of Total Loss with respect to all the weights in our neural network, its time for second step, ie, to update weights.

    w1 = w1 - learning_rate * δE_total/δw1
    w2 = w2 - learning_rate * δE_total/δw2
    w3 = w3 - learning_rate * δE_total/δw3
    w4 = w4 - learning_rate * δE_total/δw4
    w5 = w5 - learning_rate * δE_total/δw5
    w8 = w6 - learning_rate * δE_total/δw6
    w7 = w7 - learning_rate * δE_total/δw7
    w8 = w8 - learning_rate * δE_total/δw8

This completes backpropagation process for this iteration.

Now when the next datapoint or batch is fed into neural network, it will use updated weights. Feedforwward process will happen followed by backpropagation and so on. This process will continue till weights are no longer updated or loss function reaches its minimum value, at which point network training stops effectively.

Learning Rate : 0.1

![image](https://user-images.githubusercontent.com/67177106/137674930-326b8024-f44e-404a-883b-b09c86277744.png)

Learning Rate : 0.2

![image](https://user-images.githubusercontent.com/67177106/137674978-a20326c8-99b7-4d0c-8a4d-92aa326dda86.png)

Learning Rate : 0.5

![image](https://user-images.githubusercontent.com/67177106/137675119-f7e97ccb-7a05-482e-a87a-05222fa7193c.png)


Learning Rate : 0.8

![image](https://user-images.githubusercontent.com/67177106/137675161-90eae800-37b1-4de0-a605-365bf34c1b17.png)


Learning Rate: 1

![image](https://user-images.githubusercontent.com/67177106/137675216-cce3d352-3d2f-4fc5-b585-e682dd11f46e.png)

Learning Rate : 2

![image](https://user-images.githubusercontent.com/67177106/137675243-da79ed3f-6965-44c8-a5a0-4a8d02c29929.png)






## Assignment Part 2
---------------------------------

1) Number of parameters used < 20000
2) Achieved 99.4% accuracy in 17th Epoch


## Model Parameters
-----------------------

    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
              ReLU-3           [-1, 16, 26, 26]               0
         Dropout2d-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           2,304
       BatchNorm2d-6           [-1, 16, 24, 24]              32
              ReLU-7           [-1, 16, 24, 24]               0
            Conv2d-8           [-1, 32, 22, 22]           4,608
       BatchNorm2d-9           [-1, 32, 22, 22]              64
             ReLU-10           [-1, 32, 22, 22]               0
        Dropout2d-11           [-1, 32, 22, 22]               0
           Conv2d-12           [-1, 16, 22, 22]             528
             ReLU-13           [-1, 16, 22, 22]               0
        MaxPool2d-14           [-1, 16, 11, 11]               0
           Conv2d-15             [-1, 16, 9, 9]           2,304
      BatchNorm2d-16             [-1, 16, 9, 9]              32
             ReLU-17             [-1, 16, 9, 9]               0
        Dropout2d-18             [-1, 16, 9, 9]               0
           Conv2d-19             [-1, 16, 9, 9]           2,304
      BatchNorm2d-20             [-1, 16, 9, 9]              32
             ReLU-21             [-1, 16, 9, 9]               0
        Dropout2d-22             [-1, 16, 9, 9]               0
           Conv2d-23             [-1, 16, 7, 7]           2,304
      BatchNorm2d-24             [-1, 16, 7, 7]              32
             ReLU-25             [-1, 16, 7, 7]               0
        Dropout2d-26             [-1, 16, 7, 7]               0
           Conv2d-27             [-1, 32, 5, 5]           4,608
      BatchNorm2d-28             [-1, 32, 5, 5]              64
             ReLU-29             [-1, 32, 5, 5]               0
        Dropout2d-30             [-1, 32, 5, 5]               0
           Conv2d-31             [-1, 10, 5, 5]             320
        AvgPool2d-32             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 19,712
    Trainable params: 19,712
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.28
    Params size (MB): 0.08
    Estimated Total Size (MB): 1.35
    ----------------------------------------------------------------

Training Logs :


Epoch 1 : 

  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:77: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.09951310604810715 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.03it/s]


Test set: Average loss: 0.0502, Accuracy: 9858/10000 (98.58%)


Epoch 2 : 

loss=0.12439335137605667 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.00it/s]


Test set: Average loss: 0.0443, Accuracy: 9845/10000 (98.45%)


Epoch 3 : 

loss=0.11283024400472641 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.70it/s]


Test set: Average loss: 0.0350, Accuracy: 9900/10000 (99.00%)


Epoch 4 : 

loss=0.15605205297470093 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.42it/s]


Test set: Average loss: 0.0316, Accuracy: 9899/10000 (98.99%)


Epoch 5 : 

loss=0.14872819185256958 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.08it/s]


Test set: Average loss: 0.0282, Accuracy: 9905/10000 (99.05%)


Epoch 6 : 

loss=0.07366801053285599 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.44it/s]


Test set: Average loss: 0.0295, Accuracy: 9907/10000 (99.07%)


Epoch 7 : 

loss=0.07596011459827423 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.21it/s]


Test set: Average loss: 0.0281, Accuracy: 9915/10000 (99.15%)


Epoch 8 : 

loss=0.010262154042720795 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.88it/s]


Test set: Average loss: 0.0260, Accuracy: 9921/10000 (99.21%)


Epoch 9 : 

loss=0.09185423702001572 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.74it/s]


Test set: Average loss: 0.0236, Accuracy: 9921/10000 (99.21%)


Epoch 10 : 

loss=0.08244063705205917 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.40it/s]


Test set: Average loss: 0.0261, Accuracy: 9919/10000 (99.19%)


Epoch 11 : 

loss=0.011143454350531101 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.47it/s]


Test set: Average loss: 0.0234, Accuracy: 9924/10000 (99.24%)


Epoch 12 : 

loss=0.045645054429769516 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.36it/s]


Test set: Average loss: 0.0279, Accuracy: 9913/10000 (99.13%)


Epoch 13 : 

loss=0.013037172146141529 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.35it/s]


Test set: Average loss: 0.0276, Accuracy: 9920/10000 (99.20%)


Epoch 14 : 

loss=0.0418940968811512 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.60it/s]


Test set: Average loss: 0.0232, Accuracy: 9922/10000 (99.22%)


Epoch 15 : 

loss=0.034113917499780655 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.33it/s]


Test set: Average loss: 0.0212, Accuracy: 9939/10000 (99.39%)


Epoch 16 : 

loss=0.009824828244745731 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.13it/s]


Test set: Average loss: 0.0231, Accuracy: 9934/10000 (99.34%)


Epoch 17 : 

loss=0.04727235436439514 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.45it/s]


Test set: Average loss: 0.0182, Accuracy: 9940/10000 (99.40%)


Epoch 18 : 

loss=0.026582667604088783 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.12it/s]


Test set: Average loss: 0.0205, Accuracy: 9930/10000 (99.30%)


Epoch 19 : 

loss=0.03843732550740242 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.54it/s]


Test set: Average loss: 0.0189, Accuracy: 9944/10000 (99.44%)


Epoch 20 : 

loss=0.0451742447912693 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.26it/s]


Test set: Average loss: 0.0171, Accuracy: 9949/10000 (99.49%)

