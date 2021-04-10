clc
clear

min_mean = 0;
num_tests = 10000;
for i = 1:num_tests
    K = 2;
    sigma = 1;
    lambda = 1/(2*sigma^2);
    h = abs(normrnd(0,sigma,K,1) + 1j*normrnd(0,sigma,K,1)).^2;
    emp_mean = mean(h);
    theo_mean = 1/lambda;
    error = (emp_mean-theo_mean)^2;

    emp_min_h = min(h);
    min_mean = min_mean + emp_min_h;
end
min_mean = min_mean/num_tests
theo_min_henrik = 1/(K*lambda)
theo_min_paper = 1/K