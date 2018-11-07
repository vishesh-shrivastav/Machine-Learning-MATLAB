clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part A %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Preparation
rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+2); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

figure; 
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on 
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15) 
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

% Training kernel perceptron on data3
q = 2;
[alpha_values, b] = kernPercGD(data3,theclass,q);
N = size(data3,1);

% Step size of the grid
d = 0.02;

[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
min(data3(:,2)):d:max(data3(:,2)));

xGrid = [x1Grid(:),x2Grid(:)]; 
scores_perceptron = zeros(size(xGrid,1),1);

for i = 1:size(xGrid,1)
    row = xGrid(i,:);
    foo = 0;
    
    for j = 1:N
        inner_product = data3(j,:) * row';
        inner_product = inner_product + 1;
        inner_product = inner_product ^ q;
        foo = foo + (alpha_values(j) * theclass(j) * inner_product);
    end
    foo = foo + b;
    scores_perceptron(i) = foo;
end

figure;
gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
contour(x1Grid,x2Grid,reshape(scores_perceptron,size(x1Grid)),[0 0],'k');

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train SVM for data3
fit = fitcsvm(data3,theclass,'KernelFunction','polynomial','PolynomialOrder',q,'BoxConstraint',1);
[~, scores_svm] = predict(fit,xGrid);

% Plot decision boundary obtained from SVM
contour(x1Grid,x2Grid,reshape(scores_svm(:,2),size(x1Grid)),[0 0],'r');
legend({'-1','+1','originalclassboundary','perceptron','svm'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part C %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Optdigits 79 data %%%%%%%%%%%%%%%%%%%%%%%%
% Classify training data and find training error
load('optdigits79_train.txt');
train79 = optdigits79_train(:,1:end-1);
class79 = optdigits79_train(:,end);

[alpha79, b79] = kernPercGD(train79, class79, q);
training_error_rate79=0;

for i = 1:size(train79,1)
    foo_t = 0;
    row = train79(i,:);
    for j = 1:size(train79,1)
        inner_product79 = row * train79(j,:)';
        inner_product79 = inner_product79 + 1;
        inner_product79 = inner_product79 ^ q;
        foo_t = foo_t + (alpha79(j) * class79(j) * inner_product79);
    end
    
    if sign(foo_t + b79) ~= class79(i)
        training_error_rate79 = training_error_rate79 + 1;
    end
end

training_error_rate79 = training_error_rate79 / size(train79,1);
fprintf("Training error rate for optdigits79 data is %f \n", training_error_rate79 * 100);

% Classify optdigits79 test data and find test error
load('optdigits79_test.txt');
test79 = optdigits79_test(:, 1:end-1);
testclass79 = optdigits79_test(:, end);
test_error_rate79 = 0;

for i = 1:size(test79,1)
    foo_t = 0;
    row = test79(i,:);
    for j=1:size(train79,1)
        inner_product79 = row * train79(j,:)';
        inner_product79 = inner_product79 + 1;
        inner_product79 = inner_product79 ^ q;
        foo_t = foo_t + (alpha79(j) * class79(j) * inner_product79);
    end
    
    if sign(foo_t + b79) ~= testclass79(i)
        test_error_rate79 = test_error_rate79 + 1;
    end 
end

test_error_rate79 = test_error_rate79 / size(test79,1);
fprintf("Test error rate for optdigits79 data is %f\n", test_error_rate79 * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Optdigits 49 data %%%%%%%%%%%%%%%%%%%%%%%%

% Classify training data and find training error
load('optdigits49_train.txt');
train49 = optdigits49_train(:,1:end-1);
class49 = optdigits49_train(:,end);

[alpha49, b49] = kernPercGD(train49, class49, q);
train_err_rate=0;

for i=1:size(train49,1)
    foo_t = 0;    
    row = train49(i,:);
    for j = 1:size(train49,1)
        inner_product = row*train49(j,:)';
        inner_product = inner_product + 1;
        inner_product = inner_product ^ q;
        foo_t = foo_t + (alpha49(j) * class49(j) * inner_product);
    end
    
    if sign(foo_t + b49) ~= class49(i)
        train_err_rate = train_err_rate+1;
    end
        
end
train_err_rate = train_err_rate / size(train49,1);
fprintf("Training error rate for optdigits49 data is %f\n", train_err_rate * 100);

% Classify optdigits49 test data and find test error
load('optdigits49_test.txt');
test49 = optdigits49_test(:,1:end-1);
testclass49 = optdigits49_test(:,end);
test_error_rate49 = 0;

for t = 1:size(test49,1)
    foo_t = 0;
    
    row = test49(t,:);
    for i = 1:size(train49,1)
        inner_product = row * train49(i,:)';
        inner_product = inner_product + 1;
        inner_product = inner_product ^ q;
        foo_t = foo_t + (alpha49(i) * class49(i) * inner_product);
    end
    
    if sign(foo_t + b49) ~= testclass49(t)
        test_error_rate49 = test_error_rate49 + 1;
    end
        
end

test_error_rate49 = test_error_rate49 / size(test49,1);
fprintf("Test error rate for optdigits49 data is %f\n", test_error_rate49 * 100);