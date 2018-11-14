function [probab1,probab2, pc1, pc2] = Bayes_learning(training_data, validation_data)
train_classes_given = training_data(:,end);
validation_classes_given = validation_data(:, end);
minimum_error = 1;
% Subset the data to get classes 1 and 2
class_1_train = training_data(training_data(:,end)==1, :);
class_2_train = training_data(training_data(:,end)==2, :);

probab1=mean(class_1_train,1);
probab2=mean(class_2_train,1);

sigmas = [-5 -4 -3 -2 -1 0 1 2 3 4 5];

for i = 1:size(sigmas,2)
    prior1 = 1 / (1 + exp(-sigmas(i)));
    prior2 = 1 - prior1;
    assigned_classes = [];
    error = 0;
    for r = 1:size(validation_data,1)
        prod1 = prior1;
        prod2 = prior2;
        
        for j = 1:22
            prod1 = (probab1(j)^validation_data(r,j)) * ((1 - probab1(j))^(1-validation_data(r,j))) * prod1;
            prod2 = (probab2(j)^validation_data(r,j)) * ((1 - probab2(j))^(1-validation_data(r,j))) * prod2;
        end
        
        if(prod1>prod2)
            assigned_class = 1;
        else
            assigned_class = 2;
        end
        
        if(validation_classes_given(r)~=assigned_class) 
            error=error+1;
        end
    end
% Calculate error
error_rate=(error/size(validation_data,1));
disp(sprintf("Error for sigma = %d", sigmas(i)));
disp(error_rate*100);

    if(error_rate < minimum_error)
        minimum_error = error_rate;
        minimum_sigma = sigmas(i);
    end

end

fprintf("Best sigma is: %d \n", minimum_sigma);

pc1 = 1 / (1 + exp(-minimum_sigma));
pc2 = 1 - pc1;
end