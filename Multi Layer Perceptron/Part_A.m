clear;
H_vector = [3,6,9,12,15,18];
error_trains = [];
error_valid = [];
min_v = 1000;
% Calling mlptrain for each value of hidden units
for e = 1:size(H_vector,2)
    fprintf('Training for H = %d hidden units. \n',H_vector(1,e));
    [z,w,v,et,ev] = mlptrain('optdigits_train.txt', 'optdigits_valid.txt', H_vector(1,e), 10);
    error_trains(end+1) = et;
    error_valid(end+1) = ev;
    % Find best value of number of hidden units. Choose the value for which
    % validation error is lowest
    if (ev < min_v)
        min_v = ev;
        z_min = z;
        w_min = w;
        v_min = v;
        H_min = H_vector(1,e);
    end
end

% Plot training errors and validation error vs number of hidden units
figure
p1 = plot(H_vector, error_trains);
hold on
p2 = plot(H_vector, error_valid);
legend('training error','validation error');

fprintf("Lowest error rate is for %d no. of hidden units \n",H_min);

% Use this to classify test data
[z_test] = mlptest('optdigits_test.txt',w_min,v_min);