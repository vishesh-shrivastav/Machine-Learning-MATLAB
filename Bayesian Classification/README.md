Classification using Bayesian learning with prior and posterior probabilities.

First, calculate maximum likelihood estimation on the training set, "SPECT_train.txt". Then, using learned Bernoulli distributions and
the given prior function, classify the samples in the validation set ("SPECT_valid.txt") using these values for sigma: (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5).   Then, use the best prior to classify test data ("SPECT_test.txt") and report the error rate.

We consider prior to be defined as:  
![img1](https://latex.codecogs.com/gif.latex?%24P%28C_1%20%7C%20%5Csigma%29%20%3D%201/1&plus;e%5E%7B%28-%5Csigma%29%7D%24)
and  
![img2](https://latex.codecogs.com/gif.latex?%24P%28C_2%29%20%3D%201%20-%20P%28C_1%29%24)
