% Call EM algorithm for 'goldy.bmp' for k=7 clusters
try
    EMG(0, 'goldy.bmp', 7); %Fails
catch
    disp('Algorithm failed on goldy for k=7. Continuing with execution of script.');
end
% Clustering using built-in k-means function
[img, cmap] = imread('goldy.bmp');
% Convert indexed image to RGB
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);
goldy = reshape(img_double,[],3);

N = size(goldy,1);
[idx, means] = kmeans(goldy,7);

% Plot the compressed image given by kmeans

% Color of each pixel is the mean of the cluster it belongs to
% Create color values for each pixel
color_vals = zeros(N,3);
for i = 1:N
    color_vals(i,:) = means(idx(i),:);
end

% Reshape this (d1*d2) * 3 matrix to d1 * d2 * 3 matrix
compressed_image = reshape(color_vals,size(img_double,1),size(img_double,2),3);
figure('Name','Compressed image given by Kmeans')
imshow(compressed_image);

% Call improved EM algorithm for 'goldy,bmp' for k=7 clusters
[h,m,q] = EMG(1, 'goldy.bmp', 7);