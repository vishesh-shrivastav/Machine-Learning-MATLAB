function [z, w, V, error_t_rate, error_v_rate] = mlptrain(train_path, valid_path, H, k)
%train_path = 'optdigits_train.txt';
%valid_path = 'optdigits_valid.txt';
trainn = load(train_path);
valid = load(valid_path);

% Learning factor
lf = 0.00001;
% Initialise weight matrices
% w matrix - size H * (d+1), where H = no. of hidden units and d = no. of
% columns in data
a = -0.01;
b = 0.01;
w = (b-a).*rand(H,size(trainn,2)) + a;
% v matrix - size K * (H+1), where K = no. of classes and H = no. of hidden
% units
V = (b-a).*rand(10,H+1) + a;
converged = 0;
% z matrix - size n * H, where n = no. of samples and H = no. of hidden
% units
z = zeros(size(trainn,1), H);
iteration = 0;
net_error = 0;
while (converged == 0)
   old_error = net_error;
   iteration = iteration + 1;
   if iteration > 800
       lf = 10^-8;
   end
   % Iterate over the whole dataset rowwise
   for ind = 1:size(trainn,1)
      row = trainn(ind,1:end-1);
      row(:,end+1) = 1;
      
    % Populate r vector
    % If class label of row is 'l', the 'l+1'th index of r is one and
    % all others are zero
    r = zeros(1,10);
    r(trainn(ind,65)+1) = 1;
    
    % Step 1 - Populate z matrix using ReLU
    for h = 1:H
        prod1 = w(h,:) * transpose(row);
        if (prod1 >= 0)
            z(ind,h) = prod1;
        end
    end
    
    % Step 2 - Calculate the outputs of the perceptron using softmax
    % Add bias column
    z(:,H+1) = 1;
    denom = 0;
    list_of_o_i = [];
    for i = 1:10
        o_i = V(i,:) * transpose(z(ind,:));
        list_of_o_i(end + 1) = o_i;
        denom = denom + exp(o_i);
    end

    y = zeros(1,10);
    for i = 1:10
        y(1,i) = exp(list_of_o_i(i))/denom;
    end
    
   % Step 3 - Update rule for V

    delta_v = zeros(10,H+1);
    for i = 1:10
        delta_v(i,:) = lf * (r(i) - y(i)) * z(ind,:);
    end
    
    % Step 4 - Update rule for w - Add if condition
    delta_w = zeros(H, 65);
    for j = 1:65
        for h = 1:H
            sum_h = 0;
            for i = 1:10
                sum_h = sum_h + (r(i) - y(i)) * V(i,h);
            end
            delta_w(h,j) = lf * sum_h * row(1,j);
        end
    end
    
    % Step 5 - Update V
    V = V + delta_v;
    % Step 6 - Update w
    w = w + delta_w;
    
    % Calculate error of single data row
    error_row = 0;
    for i = 1:10
        error_row = error_row + (r(i) * log(y(i)));
    end
    
    % Update net error
    net_error = net_error + error_row;
    
   end
   diff = abs(net_error - old_error);
   if (iteration > 1000 || diff < 0.01)
       converged = 1;
   end
end

% Classify train data
% List of predicted classes for train data
predicted_train = [];
% z_train matrix - size n * H, where n = no. of samples and H = no. of hidden
% units
z_train = zeros(size(trainn,1), H);
for i = 1:size(trainn,1)
    % Populate z
    for h = 1:H
        prod1 = w(h,:) * transpose(trainn(i,:));
        if (prod1 >= 0)
            z_train(i,h) = prod1;
        end
    end
    
    % Calculate output
    % Add bias column
    z_train(:,H+1) = 1;
    
    denom_t = 0;
    list_of_o_i = [];
    for a = 1:10
        o_i = V(a,:) * transpose(z_train(i,:));
        list_of_o_i(end + 1) = o_i;
        denom_t = denom_t + exp(o_i);
    end

    y_train = zeros(1,10);
    for ii = 1:10
        y_train(1,ii) = exp(list_of_o_i(ii))/denom_t;
    end
    
    % Find index of max value in y_train
    [~,idxx] = max(y_train);
    % Assign (index-1) as predicted class for this row of data
    predicted_train(end + 1) = idxx-1;
end

% Calculate training error
original_classes = trainn(:,65);
pt_transpose = transpose(predicted_train);
error = 0;

for j = 1:size(pt_transpose,1)
    if (pt_transpose(j) ~= original_classes(j))
        error = error + 1;
    end
end

error_t_rate = (error/size(trainn,1)) * 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Classify Validation Data%%%%%%%%%%%%%%%%%
% Classify Validation data
% List of predicted classes for train data
predicted_valid = [];
% z_valid matrix - size n * H, where n = no. of samples and H = no. of hidden
% units
z_valid = zeros(size(valid,1), H);
for i = 1:size(valid,1)
    % Populate z
    for h = 1:H
        prod1 = w(h,:) * transpose(valid(i,:));
        if (prod1 >= 0)
            z_valid(i,h) = prod1;
        end
    end
    
    % Calculate output
    % Add bias column
    z_valid(:,H+1) = 1;
    
    denom_v = 0;
    list_of_o_i = [];
    for a = 1:10
        o_i = V(a,:) * transpose(z_valid(i,:));
        list_of_o_i(end + 1) = o_i;
        denom_v = denom_v + exp(o_i);
    end

    y_valid = zeros(1,10);
    for ii = 1:10
        y_valid(1,ii) = exp(list_of_o_i(ii))/denom_v;
    end
    
    % Find index of max value in y_train
    [~,idxx] = max(y_valid);
    % Assign (index-1) as predicted class for this row of data
    predicted_valid(end + 1) = idxx-1;
end

% Calculate training error
original_classes_v = valid(:,65);
pv_transpose = transpose(predicted_valid);
error_v = 0;

for j = 1:size(pv_transpose,1)
    if (pv_transpose(j) ~= original_classes_v(j))
        error_v = error_v + 1;
    end
end

error_v_rate = (error_v/size(valid,1)) * 100;
fprintf("Training error rate for H = %d is %d\n", H, error_t_rate);
fprintf("Validation error rate for H = %d is %d\n", H, error_v_rate);
