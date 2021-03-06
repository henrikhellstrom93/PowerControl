%% Two devices, no simplification/heuristic
clc
clear
num_av = 10;
num_optimal = 0;
av_x_error = 0;
av_sqp_error = 0;
x_errors = zeros(num_av,1);
sqp_errors = zeros(num_av,1);
a = 0;
while a < num_av
    a = a + 1;
    %Number of devices
    K = 2;
    %Rayleigh fading
    h = normrnd(0,1/2,K,1) + 1j*normrnd(0,1/2,K,1);
    %Maximum power
    P_max = 10*ones(K,1);
    %Sort h and P_max by quality indicator
    [quality_indicator, sort_index] = sort(abs(h).^2.*P_max);
    P_max_sort = zeros(K,1);
    h_sort = zeros(K,1);
    for i = 1:K
        P_max_sort(i) = P_max(sort_index(i));
        h_sort(i) = h(sort_index(i));
    end
    P_max = P_max_sort;
    h = h_sort;
    %Noise variance
    sigma_w = 1;
    %Noise
    w = randn(1,1)*sigma_w;
    %Statistics on data source
    mu_s = 1;
    sigma_s = 1;
    %Generate s
    s = randn(K,1)*sigma_s+mu_s;
    %SQP with optimized normalization
    p = optimvar('p',K,1);
    eta = optimvar('eta',1,1);
    mu_n = optimvar('mu_n',1,1);
    sigma_n = optimvar('sigma_n',1,1);
    %Noise and signal evaluated
    obj = ...
        (mu_s^2+sigma_s^2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*mu_s^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        (mu_s^2+sigma_s^2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %Optimization problem
    prob = optimproblem('Objective',obj);
    nlcons = p*(mu_n^2+sigma_n^2) <= P_max;
    lcons = p >= 0;
    prob.Constraints.circlecons = nlcons;
    prob.Constraints.linearcons = lcons;
    x0.p = ones(K,1);
    x0.eta = 1;
    x0.mu_n = 0;
    x0.sigma_n = 1;
    [sol,fval,exitflag,output] = solve(prob,x0);
    if exitflag ~= 1
        disp("Not optimal")
    else
        num_optimal = num_optimal + 1;
    end
    p = sol.p;
    eta = sol.eta;
    sigma_n = sol.sigma_n;
    mu_n = sol.mu_n;
    %Signal and noise expectation evaluated
    sqp_obj = ...
        (mu_s^2+sigma_s^2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*mu_s^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        (mu_s^2+sigma_s^2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %Calculate error
    true_sum = sum(s);
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(sqrt(p).*abs(h).*s_norm/sqrt(eta)) + w/sqrt(eta);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, K);
    sqp_error = real(true_sum-est_sum)^2;
    av_sqp_error = av_sqp_error + sqp_error;
    sqp_errors(a) = sqp_error;
    %Evaluate xiaowen objective
    [p_x, eta_x] = xiaowen_power(P_max, h, sigma_w);
    p = p_x;
    eta = eta_x;
    mu_n = 0;
    sigma_n = 1;
    x_obj = ...
        (mu_s^2+sigma_s^2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*mu_s^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        (mu_s^2+sigma_s^2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %Calculate error
    true_sum = sum(s);
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(sqrt(p).*abs(h).*s_norm/sqrt(eta)) + w/sqrt(eta);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, K);
    x_error = real(true_sum-est_sum)^2;
    av_x_error = av_x_error + x_error;
    x_errors(a) = x_error;
end
av_x_error = av_x_error/num_av;
av_sqp_error = av_sqp_error/num_av;

%%
x = 1:num_av;
figure;
hold on;
plot(x, x_errors);
plot(x, sqp_errors);
legend("xiaowen", "henrik", 'Location', 'northwest');
xlabel("Test #")
ylabel("Estimation error")