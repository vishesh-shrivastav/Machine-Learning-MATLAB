function [alpha_values, b] = kernPercGD(train_data, train_label, degree)

n = size(train_data,1);

% Construct gram matrix for train data based on polynomial kernel function
gram_matrix = train_data * train_data';
gram_matrix = arrayfun(@(x) x+1, gram_matrix);
gram_matrix = arrayfun(@(x) x^degree, gram_matrix);

% Initialise variables
alpha_values = zeros(n,1);
predicted_classes = zeros(n,1);
b = 0;
error = 0;
previous_error = 0;
has_converged = 0;
iteration = 0;

while (has_converged ~= 1)
    iteration = iteration+1;
    for t = 1:n
        foo = 0;
        for i = 1:n
            foo = foo + (alpha_values(i,1) * train_label(i) * gram_matrix(t,i));
        end
        foo = foo + b;
        if (foo * train_label(t) <= 0)
            alpha_values(t) = alpha_values(t) + 1;
            b = b + train_label(t);
            error = error + (1 - (foo * train_label(t)));
        end
        predicted_classes(t) = foo * train_label(t);
    end
    
    % Converge if error becomes zero(for toy data)
    % or if error does not change in consecutive iterations
    if(error == 0 || abs(previous_error-error) == 0)
        has_converged = 1;
    end
    
    previous_error = error;
    error = 0;
end
end