clc
clear

num_av = 100;
av_x_error = 0;
av_sqp_error = 0;
for i = 1:num_av
    %Number of devices
    K = 10;
    %Rayleigh fading
    h = normrnd(0,1/2,K,1) + 1j*normrnd(0,1/2,K,1);
    %Noise variance
    sigma_w = 1;
    %Maximum power
    P_max = 10*ones(K,1);
    %Statistics on data source
    mu_s = 1;
    sigma_s = 1;
    %SDP with optimized normalization
    b = optimvar('b',K,1);
    eta = optimvar('eta',1,1);
    mu_n = optimvar('mu_n',1,1);
    sigma_n = optimvar('sigma_n',1,1);
    obj = (mu_s^2+sigma_s^2)*sum((sqrt(b).*abs(h)/(sqrt(mu_n^2+sigma_n^2)*sqrt(eta))-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta))*sum(b.*abs(h).^2/(sqrt(eta)*(mu_n^2+sigma_n^2))-sqrt(b).*abs(h)/sqrt(mu_n^2+sigma_n^2)) + (mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta)*sum(b.*abs(h).^2/(mu_n^2+sigma_n^2)) + sigma_s^2*sigma_w^2/(sigma_n^2*eta) + K^2*(mu_s*sigma_n-mu_n*sigma_s)^2/sigma_n^2;
    prob = optimproblem('Objective',obj);
    nlcons = b <= P_max;
    prob.Constraints.circlecons = nlcons;
    show(prob)
    x0.b = ones(K,1);
    x0.eta = 1;
    x0.mu_n = 0;
    x0.sigma_n = 1;
    [sol,fval,exitflag,output] = solve(prob,x0)

    %SDP with fixed normalization
%     mu_n = mu_s*0;
%     sigma_n = sigma_s*1;
%     b = optimvar('b',K,1);
%     eta = optimvar('eta',1,1);
%     obj = (mu_s^2+sigma_s^2)*sum((sqrt(b).*abs(h)/(sqrt(mu_n^2+sigma_n^2)*sqrt(eta))-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta))*sum(b.*abs(h).^2/(sqrt(eta)*(mu_n^2+sigma_n^2))-sqrt(b).*abs(h)/sqrt(mu_n^2+sigma_n^2)) + (mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta)*sum(b.*abs(h).^2/(mu_n^2+sigma_n^2)) + sigma_s^2*sigma_w^2/(sigma_n^2*eta) + K^2*(mu_s*sigma_n-mu_n*sigma_s)^2/sigma_n^2;
%     prob = optimproblem('Objective',obj);
%     nlcons = b <= P_max;
%     prob.Constraints.circlecons = nlcons;
%     show(prob)
%     x0.b = ones(K,1);
%     x0.eta = 1;
%     [sol2,fval2,exitflag,output] = solve(prob,x0)
%     sol2.mu_n = 0;
%     sol2.sigma_n = 1;

    w = normrnd(0,sigma_w);
    s = sigma_s*randn(K,1) + mu_s;
    H = diag(h);
    n = K
    %Xiaowen solution
    [p_x, eta_x] = xiaowen(P_max, H, sigma_w);
    true_sum = sum(s);
    mu_n = 0;
    sigma_n = 1;
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(p_x.*diag(H).*s_norm/sqrt(eta_x)) + w/sqrt(eta_x);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, n);
    x_error = real(true_sum-est_sum)^2;
    av_x_error = av_x_error + x_error;
    
    fval_xiaowen = (mu_s^2+sigma_s^2)*sum((abs(p_x).*abs(h)/(sqrt(mu_n^2+sigma_n^2)*sqrt(eta_x))-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta_x))*sum(abs(p_x).^2.*abs(h).^2/(sqrt(eta_x)*(mu_n^2+sigma_n^2))-abs(p_x).*abs(h)/sqrt(mu_n^2+sigma_n^2)) + (mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta_x)*sum(abs(p_x).^2.*abs(h).^2/(mu_n^2+sigma_n^2)) + sigma_s^2*sigma_w^2/(sigma_n^2*eta_x) + K^2*(mu_s*sigma_n-mu_n*sigma_s)^2/sigma_n^2;
    %Henrik solution
    b_henrik = conj(h).*sqrt(sol.b/(sol.mu_n^2+sol.sigma_n^2))./abs(h);
    eta_henrik = sol.eta;
    true_sum = sum(s);
    mu_n = sol.mu_n;
    sigma_n = sol.sigma_n;
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(b_henrik.*diag(H).*s_norm/sqrt(eta_henrik)) + w/sqrt(eta_henrik);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, n);
    henrik_error = real(true_sum-est_sum)^2;
    av_henrik_error = av_henrik_error + henrik_error;
end
av_henrik_error/num_av-av_x_error/num_av

hold on;
plot(1:n,p_x.*diag(H)/sqrt(eta_x))
plot(1:n,b_henrik.*diag(H)/sqrt(eta_henrik))
%% Verify xiaowen solution
clc
clear
%Number of devices
K = 10;
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
%Statistics on data source
mu_s = 0;
sigma_s = 1;
mu_n = 0;
sigma_n = 1;
%SQP with optimized normalization
b = optimvar('b',K,1);
eta = optimvar('eta',1,1);
obj = (mu_s^2+sigma_s^2)*sum((sqrt(b).*abs(h)/(sqrt(mu_n^2+sigma_n^2)*sqrt(eta))-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta))*sum(b.*abs(h).^2/(sqrt(eta)*(mu_n^2+sigma_n^2))-sqrt(b).*abs(h)/sqrt(mu_n^2+sigma_n^2)) + (mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta)*sum(b.*abs(h).^2/(mu_n^2+sigma_n^2)) + sigma_s^2*sigma_w^2/(sigma_n^2*eta) + K^2*(mu_s*sigma_n-mu_n*sigma_s)^2/sigma_n^2;
prob = optimproblem('Objective',obj);
nlcons = b <= P_max;
prob.Constraints.circlecons = nlcons;
show(prob)
x0.b = ones(K,1);
x0.eta = 1;
[sol,fval,exitflag,output] = solve(prob,x0)

%Xiaowen solution
[p_x, eta_x] = xiaowen_power(P_max, h, sigma_w);
obj_sqp = sum((sqrt(sol.b).*abs(h)/sqrt(sol.eta)-1).^2)+sigma_w^2/sol.eta;
obj_x = sum((sqrt(p_x).*abs(h)/sqrt(eta_x)-1).^2)+sigma_w^2/eta_x;


p_x_relative = sqrt(abs(p_x))./sqrt(P_max);
p_sqp_relative = sqrt(abs(sol.b))./sqrt(P_max);

%% Cauchy bound solution
clc
clear
num_av = 10;
av_h_error = 0;
av_x_error = 0;
for a = 1:num_av
    %Number of devices
    K = 10;
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
    obj = (sigma_s^2+mu_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)*sum(p.*abs(h).^2/sqrt(eta)-sqrt(p).*abs(h))+(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta*sum(p.*abs(h).^2)+sigma_s^2*sigma_w^2/sigma_n^2/eta+K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    prob = optimproblem('Objective',obj);
    nlcons = p*(mu_n^2+sigma_n^2) <= P_max;
    prob.Constraints.circlecons = nlcons;
    %prob.Constraints.linearcons = poscons;
    show(prob)
    x0.p = ones(K,1);
    x0.eta = 1;
    x0.mu_n = 0;
    x0.sigma_n = 1;
    [sol,fval,exitflag,output] = solve(prob,x0);
    p = sol.p;
    eta = sol.eta;
    mu_n = sol.mu_n;
    sigma_n = sol.sigma_n;
    fval_h = (sigma_s^2+mu_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)*sum(p.*abs(h).^2/sqrt(eta)-sqrt(p).*abs(h))+(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta*sum(p.*abs(h).^2)+sigma_s^2*sigma_w^2/sigma_n^2/eta+K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;

    true_sum = sum(s);
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(s_norm.*abs(h).*sqrt(p))+w/sqrt(eta);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, K);
    error_h = (true_sum-est_sum)^2;
    if abs(imag(error_h)) > 0
        error_h
    end
    av_h_error = av_h_error + error_h;
    
    %Xiaowen solution
    [p_x, eta_x] = xiaowen_power(P_max, h, sigma_w);
    p = p_x;
    eta = eta_x;
    mu_n = 0;
    sigma_n = 1;
    fval_x = (sigma_s^2+mu_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + 2*mu_s*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)*sum(p.*abs(h).^2/sqrt(eta)-sqrt(p).*abs(h))+(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta*sum(p.*abs(h).^2)+sigma_s^2*sigma_w^2/sigma_n^2/eta+K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;

    true_sum = sum(s);
    s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
    y = sum(s_norm.*abs(h).*sqrt(p))+w/sqrt(eta);
    est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, K);
    error_x = (true_sum-est_sum)^2;
    av_x_error = av_x_error + error_x;
    
    p_x_relative = sqrt(abs(p_x))./sqrt(P_max);
    p_sqp_relative = sqrt(abs(sol.p))./sqrt(P_max);
end
av_h_error = av_h_error/num_av;
av_x_error = av_x_error/num_av;
%% mu_s*sigma_n = mu_n*sigma_s solution
clc
clear
num_av = 1;
av_h_error = 0;
av_x_error = 0;
for a = 1:num_av
    %Number of devices
    K = 10;
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
    obj = (mu_s^2+sigma_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + sigma_s^2*sigma_w^2/sigma_n^2/eta;
    prob = optimproblem('Objective',obj);
    nlcons = p*(mu_n^2+sigma_n^2) <= P_max;
    lcons = mu_s*sigma_n == mu_n*sigma_s;
    prob.Constraints.circlecons = nlcons;
    prob.Constraints.linearcons = lcons;
    show(prob)
    x0.p = ones(K,1);
    x0.eta = 1;
    x0.mu_n = 0;
    x0.sigma_n = 1;
    [sol,fval,exitflag,output] = solve(prob,x0);
    p = sol.p;
    eta = sol.eta;
    sigma_n = sol.sigma_n;
    obj_sqp = (mu_s^2+sigma_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + sigma_s^2*sigma_w^2/sigma_n^2/eta;
    [b_norm, eta_norm, mu_n, sigma_n] = henrik_norm(P_max, diag(h), sigma_w, mu_s, sigma_s)
    p = abs(b_norm).^2;
    eta = eta_norm;
    obj_h = (mu_s^2+sigma_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + sigma_s^2*sigma_w^2/sigma_n^2/eta;
end
%% Two devices, no simplification/heuristic
clc
clear
num_av = 1;
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
    %Massive objective function for 2 devices
%     obj = (mu_s^2+sigma_s^2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+ ...
%     2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta)))+ ...
%     2*mu_s^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+ ...
%     2*mu_s*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta)))+ ...
%     2*mu_s*K*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+ ...
%     p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta)+ ...
%     2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta)))+ ...
%     2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n^2*eta)- ...
%     2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*sqrt(eta))+ ...
%     (mu_s^2+sigma_s^2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+ ...
%     2*mu_s*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/(sigma_n*sqrt(eta)))+ ...
%     2*mu_s*K*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+ ...
%     p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*eta)- ...
%     2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/(sigma_n^2*sqrt(eta))+ ...
%     sigma_s^2*sigma_w^2/(sigma_n^2*eta)+ ...
%     K^2*(mu_s*sigma_n-mu_n*sigma_s)^2/sigma_n^2;
    %No evaluation of expectation obj
%     obj = ...
%         s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
%         2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
%         2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
%         2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
%         s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
%         2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
%         2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
%         p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
%         2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
%         p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
%         2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
%         sigma_s^2*w^2/sigma_n^2/eta+...
%         2*K*sigma_s*w*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n^2/sqrt(eta)+...
%         K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %Noise evaluated
%     obj = ...
%         s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
%         2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
%         2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
%         s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
%         2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
%         2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
%         p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
%         p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
%         2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
%         sigma_s^2*sigma_w^2/sigma_n^2/eta+...
%         K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
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
    show(prob)
    x0.p = ones(K,1);
    x0.eta = 1;
    x0.mu_n = 0;
    x0.sigma_n = 0.0001;
    options = optimoptions('fmincon','MaxIter',10000);
    [sol,fval,exitflag,output] = solve(prob,x0,'Options',options);
    if exitflag ~= 1
        a = a - 1;
        disp("Not optimal, skipping this round")
        
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
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %No expectation taken
    sqp_obj2 = (sum(s.*((sqrt(p).*abs(h))/sqrt(eta)-1)+sqrt(p).*abs(h)*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta))+...
        sigma_s*w/sigma_n/sqrt(eta)+...
        K*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n)^2;
    %No expectation but expression expanded
    sqp_obj3 = ...
        s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*w^2/sigma_n^2/eta+...
        2*K*sigma_s*w*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n^2/sqrt(eta)+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    sqp_noise_terms = ...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta+...
        2*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta+...
        2*K*sigma_s*w*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n^2/sqrt(eta);
    sqp_noise_diff = -sigma_s^2*w^2/sigma_n^2/eta+sigma_s^2*sigma_w^2/sigma_n^2/eta;
    %Expectation taken for w
    sqp_obj4 = ...
        s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    %Expectation taken for s and w
    sqp_obj5 = ...
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
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    x_obj2 = (sum(s.*((sqrt(p).*abs(h))/sqrt(eta)-1)+sqrt(p).*abs(h)*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta))+...
        sigma_s*w/sigma_n/sqrt(eta)+...
        K*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n)^2;
    x_obj3 = ...
        s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sigma_s*w/sigma_n/sqrt(eta)+...
        2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)*sigma_s*w/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*w^2/sigma_n^2/eta+...
        2*K*sigma_s*w*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n^2/sqrt(eta)+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    x_obj4 = ...
        s(1)^2*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)^2+...
        2*s(1)*s(2)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*s(1)*(sqrt(p(1))*abs(h(1))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        s(2)^2*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)^2+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)/sigma_n/sqrt(eta)+...
        2*K*s(2)*(sqrt(p(2))*abs(h(2))/sqrt(eta)-1)*(mu_s*sigma_n-mu_n*sigma_s)/sigma_n+...
        p(1)*abs(h(1))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta+...
        2*sqrt(p(1)*p(2))*abs(h(1))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(1))*abs(h(1))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        p(2)*abs(h(2))^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/eta-...
        2*K*sqrt(p(2))*abs(h(2))*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2/sqrt(eta)+...
        sigma_s^2*sigma_w^2/sigma_n^2/eta+...
        K^2*(mu_n*sigma_s-mu_s*sigma_n)^2/sigma_n^2;
    x_obj5 = ...
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
%     [b_norm, eta_norm, mu_n, sigma_n] = henrik_norm(P_max, diag(h), sigma_w, mu_s, sigma_s)
%     p = abs(b_norm).^2;
%     eta = eta_norm;
%     obj_h = (mu_s^2+sigma_s^2)*sum((sqrt(p).*abs(h)/sqrt(eta)-1).^2) + sigma_s^2*sigma_w^2/sigma_n^2/eta;
end
av_x_error = av_x_error/num_optimal;
av_sqp_error = av_sqp_error/num_optimal;


x = 1:num_av;
figure;
hold on;
plot(x, x_errors);
plot(x, sqp_errors);
legend("xiaowen", "henrik", 'Location', 'northwest');
xlabel("Test #")
ylabel("Estimation error")