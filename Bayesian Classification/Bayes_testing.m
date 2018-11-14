function [] = Bayes_testing(test_data, p1, p2, pc1, pc2)
test_classes_given = test_data(:,end);
error=0;
for r = 1:size(test_data,1)
        prod1 = pc1;
        prod2 = pc2;
        for j = 1:22
            prod1 = (p1(j)^test_data(r,j)) * ((1 - p1(j))^(1-test_data(r,j))) * prod1;
            prod2 = (p2(j)^test_data(r,j)) * ((1 - p2(j))^(1-test_data(r,j))) * prod2;
        end
        
        if (prod1 > prod2)
           assigned_class=1;
        else
            assigned_class=2;
        end
         
        if (test_classes_given(r)~=assigned_class) 
            error=error+1;
        end
end

% Calculate error
error_rate=(error/size(test_data,1));
disp("Error on test data:");
disp(error_rate*100);
end