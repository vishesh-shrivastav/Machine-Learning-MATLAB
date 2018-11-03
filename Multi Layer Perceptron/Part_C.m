% Load test data
te = load('optdigits_test.txt');
class_labels_test = te(:,end);
% Add bias column
te(:,65) = 1;
% Calculate Z_test - values of the hidden layer - size N * H_min where N is the
% total no. of samples of test data
Z_test = zeros(size(te,1), H_min);
for i = 1:size(te,1)
    for h = 1:H_min
        prod1 = w_min(h,:) * transpose(te(i,:));
        if (prod1 >= 0)
            Z_test(i,h) = prod1;
        end
    end
end

% Project Z_test using A to get ztest_dash of size - N * 2
ztest_dash = Z_test * A;

% Visualise z1_dash_2
figure
%subplot(2,1,1);
gscatter(ztest_dash(:,1), ztest_dash(:,2),class_labels_test);
title('Test data projected along two principal components');

% Printing class labels for 10 data points for every class
projected_data_with_class = [ztest_dash class_labels_test];
for i = 0:9
    % Select all points in class i
    c = projected_data_with_class(projected_data_with_class(:,3)==i,1:2);
    % Create vector of random indices in class i
    random_indices = randsample(1:length(c),10);
    % Subset data at these indices
    c_print = c(random_indices,:);
    text(c_print(:,1), c_print(:,2), string(i));
end

% Projecting in three dimensions
% Project Z_test using B to get ztest_dash_3 of size - N * 3
ztest_dash_3 = Z_test * B;
ztest_dash_3 = [ztest_dash_3 class_labels_test];

% Visualise ztest_dash_3
figure
scatter3(ztest_dash_3(:,1), ztest_dash_3(:,2), ztest_dash_3(:,3), 20, class_labels_test);
colormap(jet(10));
for i=0:9
    class_i=ztest_dash_3(ztest_dash_3(:,4)==i,1:3);
    randind = randsample(1:length(class_i),10);
    class_i_print=class_i(randind,:);
    text(class_i_print(:,1),class_i_print(:,2),class_i_print(:,3),(string(i)));
end

title('Test data projected along three principal components');