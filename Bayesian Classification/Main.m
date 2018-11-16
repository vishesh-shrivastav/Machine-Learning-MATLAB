train = load('SPECT_train.txt');
valid = load('SPECT_valid.txt');
test = load('SPECT_test.txt');

% Find parameters for best value of sigma
[p1,p2,pc1,pc2]=Bayes_learning(train,valid);
% Apply these parameters to classify test data
Bayes_testing(test,p1,p2,pc1,pc2);
