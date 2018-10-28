Using the EM algorithm for Gaussian mixture models to cluster the pixels of an image and compress it. Clustering done for k = 4, 8 and 12. 
The EM algorithm is terminated when complete log likelihood ratios for Expectation step and Maximization converge. Error handling is done for  
cases when the covariance matrix becomes singular. A regularization term is added to account for singularity.
