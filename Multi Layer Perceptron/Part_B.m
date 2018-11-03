% Combine train and validation data
t = load('optdigits_train.txt');
v = load('optdigits_valid.txt');

combined = [t;v];
class_labels = combined(:,end);
% Add bias column
combined(:,65) = 1;

% Calculate Z1 - values of the hidden layer - size N * H_min where N is the
% total no. of samples of combined data
z1 = zeros(size(combined,1), H_min);
for i = 1:size(combined,1)
    for h = 1:H_min
        prod1 = w_min(h,:) * transpose(combined(i,:));
        if (prod1 >= 0)
            z1(i,h) = prod1;
        end
    end
end

% Apply pca with 2 principal components to z_one; obtain projection matrix
% A of size - H_min * 2
PCA_z1 = pca(z1);
A = PCA_z1(:,1:2);

% Project Z1 using A to get z_one_dash of size - N * 2
z1_dash_2 = z1 * A;

% Visualise z1_dash_2
figure
%subplot(2,1,1);
gscatter(z1_dash_2(:,1), z1_dash_2(:,2),class_labels);
title('Combined data projected along two principal components');

% Printing class labels for 10 data points for every class
projected_data_with_class = [z1_dash_2 class_labels];
for i = 0:9
    % Select all points in class i
    c = projected_data_with_class(projected_data_with_class(:,3)==i,1:2);
    % Create vector of random indices in class i
    random_indices = randsample(1:length(c),10);
    % Subset data at these indices
    c_print = c(random_indices,:);
    text(c_print(:,1), c_print(:,2), string(i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply pca with 3 principal components to z_one; obtain projection matrix
% A of size - H_min * 3
B = PCA_z1(:,1:3);

% Project Z1 using A to get z_one_dash of size - N * 3
z1_dash_3 = z1 * B;
z1_dash_3 = [z1_dash_3 class_labels];

% Visualise z1_dash_3
figure
scatter3(z1_dash_3(:,1), z1_dash_3(:,2), z1_dash_3(:,3), 20, class_labels);

colormap(jet(10));
for i=0:9
    class_i=z1_dash_3(z1_dash_3(:,4)==i,1:3);
    randind = randsample(1:length(class_i),10);
    class_i_print=class_i(randind,:);
    text(class_i_print(:,1),class_i_print(:,2),class_i_print(:,3),(string(i)));
end

title('Combined data projected along three principal components');