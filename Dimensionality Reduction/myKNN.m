function [error_rate, predicted_classes] = myKNN(train_data, test_data, k)
    train_classes = train_data(:,end);
    test_classes = test_data(:,end);
    train_data_wo_class = train_data(:,1:end-1);
    test_data_wo_class = test_data(:,1:end-1);
    predicted_classes = [];
    error = 0;
    for i = 1:(size(test_data,1))
        dist_and_class = zeros(size(train_data,2),2);
        for j = 1:size(train_data,1)
            %dist_ij = pdist2(test_data_wo_class(i,:), train_data_wo_class(j,:));
            dist_ij = norm(test_data_wo_class(i,:) - train_data_wo_class(j,:));
            dist_and_class(j,1) = dist_ij;
            dist_and_class(j,2) = train_classes(j);
        end
        dist_and_class_sorted = sortrows(dist_and_class,1);
        top_k_classes = dist_and_class_sorted(1:k,2);
        assigned_class = mode(top_k_classes);
        predicted_classes(end + 1) = assigned_class;
        if (assigned_class ~= test_classes(i))
            error = error + 1;
        end
    end
predicted_classes = transpose(predicted_classes);
% Error rate
error_rate = (error/size(test_data,1)) * 100;
end