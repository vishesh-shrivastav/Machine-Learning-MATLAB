function [principal_components, eigen_values] = myPCA(data)
    % Find covariance matrix of the train data
    cov_data = cov(data(:,1:end-1));
    % Calculate eigenvectors and eigenvalues of the covariance matrix
    [V_data, D_data] = eig(cov_data);
    % Variances of the components
    [eigen_values, indices] = sort(diag(D_data),'descend');
    % Sort eigenvectors according to eigenvalues
    principal_components = V_data(:, indices);
end