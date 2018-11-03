function [z] = mlptest(test, w, v)
test = load(test);
% Test data without class labels
data_test = test(:,1:64);
% Add bias column
data_test(:,65) = 1;
% No. of hidden units
H = size(v,2) - 1;
% Classify Test data
% List of predicted classes for test data
predicted_test = [];
% z_valid matrix - size n * H, where n = no. of samples and H = no. of hidden
% units
z = zeros(size(data_test,1), H);
for i = 1:size(data_test,1)
    % Populate z
    for h = 1:H
        prod1 = w(h,:) * transpose(data_test(i,:));
        if (prod1 >= 0)
            z(i,h) = prod1;
        end
    end
    
    % Calculate output
    % Add bias column
    z(:,H+1) = 1;
    
    denom_v = 0;
    list_of_o_i = [];
    for a = 1:10
        o_i = v(a,:) * transpose(z(i,:));
        list_of_o_i(end + 1) = o_i;
        denom_v = denom_v + exp(o_i);
    end

    y_test = zeros(1,10);
    for ii = 1:10
        y_test(1,ii) = exp(list_of_o_i(ii))/denom_v;
    end
    
    % Find index of max value in y_train
    [~,idxx] = max(y_test);
    % Assign (index-1) as predicted class for this row of data
    predicted_test(end + 1) = idxx-1;
end

% Calculate training error
original_classes_test = test(:,65);
pt_transpose = transpose(predicted_test);
error = 0;

for j = 1:size(pt_transpose,1)
    if (pt_transpose(j) ~= original_classes_test(j))
        error = error + 1;
    end
end

etest = (error/size(test,1)) * 100;
fprintf("Test error rate for H = %d is %d\n", H, etest);
end