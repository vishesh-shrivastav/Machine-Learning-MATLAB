function [projection_matrix, evectors_new, eigen_values] = myLDA(data, L)

    % Cell array to store data of all classes
    class_wise_data = {};

    % Matrix to store means of all classes
    means = [];

    for i = 0:9
        class_i = data(data(:,end) == i, :);
        mean_i = mean(class_i(:,1:end-1));
        means(i+1,:) = mean_i;
        class_wise_data{1,i+1} = i+1;
        class_wise_data{2, i+1} = class_i;
    end

    % Compute within-class and between-class scatter matrices
    % Within class scatter matrix
    S_w = zeros(64);
    for xx = 1:10
        S_class = zeros(64);
        for yy = 1:size(class_wise_data{2,xx})
            row = transpose(class_wise_data{2,xx}(yy,1:64));
            mean_class = transpose(means(xx,:));
            diff = row - mean_class;
            prod = diff * transpose(diff);
            S_class = S_class + prod;
        end
    S_w = S_w + S_class;
    end

    % Between class scatter matrix
    % Calculate overall mean by calculating means for each class
    overall_mean = mean(means);
    ovm = transpose(overall_mean);

    S_b = zeros(64);
    for k = 1:10
        size_of_class = size(class_wise_data{2,k},1);
        mean_c = transpose(means(k,:));
        diff_means = mean_c - ovm;
        prod_1 = size_of_class * diff_means * transpose(diff_means);
        S_b = S_b + prod_1;
    end

    % Multiply S_w inverse and S_b

    % Take pseudoinverse of S_w
    S_w_pseudo = pinv(S_w);
    % Multiply by S_b
    res = S_w_pseudo*S_b;

    % Calculate eigenvectors and eigenvalues of res
    [evectors, evalues] = eig(res);

    % Sort eigenvalues
    [evalues_sorted, indices] = sort(diag(evalues),'descend');

    % Sort eigenvectors according to eigenvalues
    eigen_values = evectors(:, indices);
    evectors_new = eigen_values(:,1:L);
    projection_matrix = data(:,1:64) * evectors_new;