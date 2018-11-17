Classification using Bayesian learning with prior and posterior probabilities.

First, calculate maximum likelihood estimation on the training set, "SPECT_train.txt". Then, using learned Bernoulli distributions and
the given prior function, classify the samples in the validation set ("SPECT_valid.txt") using these values for sigma: (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5).   Then, use the best prior to classify test data ("SPECT_test.txt") and report the error rate.

We consider prior to be defined as: $P(C_1 | \sigma) = 1/1+exp(-\sigma)$ and $P(C_2) = 1 - P(C_1)$
