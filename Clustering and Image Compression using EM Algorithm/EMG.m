function [tr_resp, mu_values, cll_m_list_t] = EMG(flag, imag, k)
%clear;
[img, cmap] = imread(imag);

% Convert indexed image to RGB
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);

data = reshape(img_double,[],3);
N = size(data,1);

% Initialisation
% Set inital priors probability values as 1/k
pi_values = zeros(k,1);
pi_values(:) = 1/k;
% Obtain initial means using kmeans
[idx,mu_values] = kmeans(data,k,'MaxIter',3,'EmptyAction','singleton');
% Initialise covariances
sigma_values = zeros(size(data,2),size(data,2),k);
for i = 1:k
    sigma_values(:,:,i) = cov(data(idx==i,:));
end

has_converged = 0;
iteration = 0;

cll_e_list = [];
cll_m_list = [];

while (has_converged ~= 1 && iteration <= 100)
    iteration = iteration + 1;
    disp('Iteration: ');
    disp(iteration);
    
    % E step
    numerators = zeros(k,N);
    denominators = zeros(N,1);
    try
    for j = 1:k
        for i = 1:N
            numerators(j,i) = pi_values(j) * mvnpdf(data(i,:), mu_values(j,:), sigma_values(:,:,j));
            % Replace zeros in the numerators matrix to small values
            if (numerators(j,i) == 0)
                numerators(j,i) =eps;
            end
            denominators(i) = denominators(i) + numerators(j,i);
        end
    end
    catch
        error('Algorithm failed since Sigma is not positive-definite. Please restart.');
        %break;
    end
    
    % Calculate the responsibilities
    responsibilities = zeros(k,N);
    Nk = zeros(k,1);
    for j = 1:k
        for i = 1:N
            responsibilities(j,i) = numerators(j,i)/denominators(i);
        end
        Nk(j) = sum(responsibilities(j,:));
    end
    
    % Complete log likelihood after E step
    cll_e = 0;
    for i = 1:N
        sum_i = 0;
        for j = 1:k
            sum_i = sum_i + responsibilities(j,i) * (log(pi_values(j)) + log(numerators(j,i)));
        end
        cll_e = cll_e + sum_i;
    end
    % Add this value to the list of complete log likelihoods
    % after E step
    cll_e_list(end + 1) = cll_e;
    
    % M Step
    lambda = 0.001;
    % Re-calculate pi, mu and sigma based on the responsibilities obtained
    for j = 1:k
        % Re-calculating pi values
        pi_values(j) = Nk(j)/N;

        % Re-calculating mu values
        sum_mu = zeros(1,3);
        for i = 1:N
            sum_mu = sum_mu + responsibilities(j,i).* data(i,:);
        end
        mu_values(j,:) = sum_mu/Nk(j);

        % Re-calculating sigma values
        sum_sigma = zeros(3,3);
        for i = 1:N
            xn = data(i,:);
            mu_k = mu_values(k,:);
            sum_sigma = sum_sigma + (responsibilities(j,i).* transpose(xn - mu_k) * (xn - mu_k));
        end
        sigma_values(:,:,j) = sum_sigma/Nk(j);
        if (flag == 1)
            sigma_values(:,:,j) = sigma_values(:,:,j) + (lambda/Nk(j)) * eye(3);
        end
    end
    
    if (flag == 1)
    % Regularization term
        tot = 0;
        for foo = 1:k
            summ = 0;
            inv_sig = inv(sigma_values(:,:,k));
            for bar = 1:size(data,2)
                summ = summ + inv_sig(bar,bar);
            end
        tot = tot + summ;
        end
        reg_term = -(lambda/2) * tot;
    end
    
    % Complete log likelihood after M step
    cll_m = 0;
    new_numerators = zeros(k,N);
    % Populate new_numerators
    try
    for j = 1:k
        for i = 1:N
                new_numerators(j,i) = pi_values(j) * mvnpdf(data(i,:), mu_values(j,:), sigma_values(:,:,j));
                % Replace zeros in the new_numerators matrix to small values
                if (new_numerators(j,i) == 0)
                    new_numerators(j,i) = eps;
                end
        end
    end
    catch
        error('Algorithm failed since Sigma is not positive-definite. Please restart.');
        %break;
    end
    
    % Calculate complete log likelihood
    for i = 1:N
        sum_i = 0;
        for j = 1:k
            sum_i = sum_i + responsibilities(j,i) * (log(pi_values(j)) + log(new_numerators(j,i)));
        end
        cll_m = cll_m + sum_i;
    end
    
    if (flag == 1)
        cll_m = cll_m + reg_term;
    end
    % Add this value to the list of complete log likelihoods
    % after M step
    cll_m_list(end + 1) = cll_m;

    % Check for convergence
    if (iteration == 1)
        err = 1000;
    else    
        err = abs(cll_m_list(iteration) - cll_m_list(iteration-1));
    end
    
    if (err <= 0.1) 
        has_converged = 1;
    end
end

%disp('h: ');
%disp(transpose(responsibilities));
%disp('m: ');
%disp(mu_values);
%disp('Q: ');
%disp(transpose(cll_m_list));
tr_resp = transpose(responsibilities);
cll_m_list_t = transpose(cll_m_list);
% Plot the compressed image

% For every pixel, the cluster to which it belongs is the cluster
% for which it has maximum value of responsibility

cluster_indexes = zeros(N,1);
for i = 1:N
    [val, idxx] = max(responsibilities(:,i));
    cluster_indexes(i) = idxx;
end

% Color of each pixel is the mean of the cluster it belongs to
% Create color values for each pixel
color_vals = zeros(N,3);
for i = 1:N
    color_vals(i,:) = mu_values(cluster_indexes(i),:);
end

% Reshape this (d1*d2) * 3 matrix to d1 * d2 * 3 matrix
figure
compressed_image = reshape(color_vals,size(img_double,1),size(img_double,2),3);
imshow(compressed_image);

% Plot complete log likelihood after every E step and M step
figure
xx = linspace(1,size(cll_m_list,2),size(cll_m_list,2));
scatter(xx,cll_m_list,'filled');
hold on;
yy = linspace(1,size(cll_e_list,2),size(cll_e_list,2));
scatter(yy,cll_e_list,'filled');

end
