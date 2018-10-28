train = load('training_data.txt');
test = load('test_data.txt');

% Subset the data to get classes 1 and 2
class_1_train = train(train(:,9)==1, :);
class_2_train = train(train(:,9)==2, :);

% Calculate mean of class 1
mu1 = mean(class_1_train(:,1:8));
% Calculate mean of class 2
mu2 = mean(class_2_train(:,1:8));

% Part a - sigma1 and sigma2 are independent

% Calculate covariance matrix of class 1
sigma1 = cov(class_1_train(:,1:8));
% Calculate covariance matrix of class 2
sigma2 = cov(class_2_train(:,1:8));

% Discriminant function calculation

% Vector containing predicted classes for each row of test data
predicted_classes = [];

% Iterate over every row of test data
% Calculate discriminant for the row for both the classes
% Assign it to the class which has the greter value for discriminant
for i = 1:size(test,1)
    row = test(i, 1:(size(test,2)-1));
    g_1 = (-1/2) * log(det(sigma1)) - (1/2) * (row - mu1) * inv(sigma1) * transpose((row - mu1)) + log(0.6);
    g_2 = (-1/2) * log(det(sigma2)) - (1/2) * (row - mu2) * inv(sigma2) * transpose((row - mu2)) + log(0.4);
    if (g_1 >= g_2)
        row_class = 1;
    else
        row_class = 2;
    end
    predicted_classes(end + 1) = row_class;
end

% Finding error
original_classes = test(:,9);
pc_transpose = transpose(predicted_classes);
error = 0;

for j = 1:size(pc_transpose,1)
    if (pc_transpose(j) ~= original_classes(j))
        error = error + 1;
    end
end

error_a_rate = (error/size(test,1)) * 100;

% Part b - sigma1 = sigma2, learned from data from both classes

% Covariance matrix calculated from both the classes
sigma = 0.6 * sigma1 + 0.4 * sigma2;

% Discriminant function calculation

% Vector containing predicted classes for each row of test data
predicted_classes_b = [];

% Iterate over every row of test data
% Calculate discriminant for the row for both the classes
% Assign it to the class which has the greter value for discriminant
for i = 1:size(test,1)
    row = test(i, 1:(size(test,2)-1));
    g_1 = (-1/2) * log(det(sigma)) - (1/2) * (row - mu1) * inv(sigma) * transpose((row - mu1)) + log(0.6);
    g_2 = (-1/2) * log(det(sigma)) - (1/2) * (row - mu2) * inv(sigma) * transpose((row - mu2)) + log(0.4);
    if (g_1 > g_2)
        row_class = 1;
    else
        row_class = 2;
    end
    predicted_classes_b(end + 1) = row_class;
end

% Finding error
pc_transpose_b = transpose(predicted_classes_b);
error_b = 0;

for j = 1:size(pc_transpose_b,1)
    if (pc_transpose_b(j) ~= original_classes(j))
        error_b = error_b + 1;
    end
end

error_b_rate = (error_b/size(test,1)) * 100;

% Part c - sigma1 and sigma2 are diagonal

% Set all off-diagonal elements of the covariance matrix to zero
sigma1_diag = diag(diag(sigma1));
sigma2_diag = diag(diag(sigma2));

% Discriminant function calculation

% Vector containing predicted classes for each row of test data
predicted_classes_c = [];

% Iterate over every row of test data
% Calculate discriminant for the row for both the classes
% Assign it to the class which has the greter value for discriminant
for i = 1:size(test,1)
    row = test(i, 1:(size(test,2)-1));
    g_1 = (-1/2) * log(det(sigma1_diag)) - (1/2) * (row - mu1) * inv(sigma1_diag) * transpose((row - mu1)) + log(0.6);
    g_2 = (-1/2) * log(det(sigma2_diag)) - (1/2) * (row - mu2) * inv(sigma2_diag) * transpose((row - mu2)) + log(0.4);
    if (g_1 > g_2)
        row_class = 1;
    else
        row_class = 2;
    end
    predicted_classes_c(end + 1) = row_class;
end

% Finding error
pc_transpose_c = transpose(predicted_classes_c);
error_c = 0;

for j = 1:size(pc_transpose_c,1)
    if (pc_transpose_c(j) ~= original_classes(j))
        error_c = error_c + 1;
    end
end

error_c_rate = (error_c/size(test,1)) * 100;

% Part d
% We relace our diagonal matrix with a diagonal matrix 
% made up of the mean of the diagonal elements
sigma1_diag_mean = trace(sigma1_diag)/size(sigma1_diag,1);
sigma2_diag_mean = trace(sigma2_diag)/size(sigma2_diag,1);

sigma1_d = diag(ones(8,1) * sigma1_diag_mean);
sigma2_d = diag(ones(8,1) * sigma2_diag_mean);

% Discriminant function calculation

% Vector containing predicted classes for each row of test data
predicted_classes_d = [];

% Iterate over every row of test data
% Calculate discriminant for the row for both the classes
% Assign it to the class which has the greter value for discriminant
for i = 1:size(test,1)
    row = test(i, 1:(size(test,2)-1));
    g_1 = (-1/2) * log(det(sigma1_d)) - (1/2) * (row - mu1) * inv(sigma1_d) * transpose((row - mu1)) + log(0.6);
    g_2 = (-1/2) * log(det(sigma2_d)) - (1/2) * (row - mu2) * inv(sigma2_d) * transpose((row - mu2)) + log(0.4);
    if (g_1 > g_2)
        row_class = 1;
    else
        row_class = 2;
    end
    predicted_classes_d(end + 1) = row_class;
end

% Finding error
pc_transpose_d = transpose(predicted_classes_d);
error_d = 0;

for j = 1:size(pc_transpose_d,1)
    if (pc_transpose_d(j) ~= original_classes(j))
        error_d = error_d + 1;
    end
end

error_d_rate = (error_d/size(test,1)) * 100;

% Printing the results
disp("Mean of class 1, mu1:");
disp(mu1);
disp("Mean of class 2, mu2:");
disp(mu2);
disp("Part A results:");
disp("S1:");
disp(sigma1);
disp("S2:");
disp(sigma2);
disp(sprintf("Part A Error Rate: %d", error_a_rate));
fprintf("\n");

disp("Part B results:");
disp("S1:");
disp(sigma);
disp("S2:");
disp(sigma);
disp(sprintf("Part B Error Rate: %d", error_b_rate));
fprintf("\n");

disp("Part C results:");
disp("S1:");
disp(sigma1_diag);
disp("S2:");
disp(sigma2_diag);
disp(sprintf("Part C Error Rate: %d", error_c_rate));
fprintf("\n");

disp("Part D results:");
disp("Alpha1:");
disp(sigma1_diag_mean);
disp("Alpha2:");
disp(sigma2_diag_mean);
disp(sprintf("Part D Error Rate: %d", error_d_rate));