%%Simulate
clc
clear
num_tests = 20;
num_optimal = 0;
s_sums = zeros(num_tests,1);
x_errors = zeros(num_tests,1);
x_errors_noisefree = zeros(num_tests,1);
h_errors = zeros(num_tests,1);
hd_errors = zeros(num_tests,1);
hd_errors_noisefree = zeros(num_tests,1);
noises = zeros(num_tests, 1);
%Number of devices
K = 20;
%Statistics on data source
mu_s = 0;
sigma_s = 1;
%Simulation specific
num_av = 1000;
%Number transmissions
T = 8;
%Maximum power
P_max = 10*ones(K,1);


for b = 1:num_tests
    av_x_error = 0;
    av_x_error_noisefree = 0;
    av_h_error = 0;
    av_hd_error = 0;
    av_hd_error_noisefree = 0;
    av_noise = 0;
    %Noise variance
    sigma_w = 0.1*b;
    for a = 1:num_av
        %Generate s
        s = randn(K,1)*sigma_s+mu_s;
        y_x = 0;
        y_x_noisefree = 0;
        y_h = 0;
        y_hd = 0;
        y_hd_noisefree = 0;
        noise = 0;
        est_sum_h = 0;
        old_h = zeros(K, T);
        old_p = zeros(K, T);
        old_eta = zeros(T, 1);
        %Rayleigh fading
        h = normrnd(0,1/2,K,1) + 1j*normrnd(0,1/2,K,1);
        for t = 1:T
            old_h(:,t) = h;
            %Sort h and P_max by quality indicator
            [quality_indicator, sort_index] = sort(abs(h).^2.*P_max);
            h_sort = zeros(K,1);
            for i = 1:K
                h_sort(i) = h(sort_index(i));
            end
            %Noise
            w = randn(1,1)*sigma_w;
            %XIAOWEN
            [p_x, eta_x] = xiaowen_power(P_max, h, h_sort, sigma_w);
            p = p_x;
            eta = eta_x;
            %Calculate receive signal
            y_x = y_x + sum(sqrt(p).*abs(h).*s/sqrt(eta)) + w/sqrt(eta);
            %y_x = sum(sqrt(p).*abs(h).*s/sqrt(eta)) + w/sqrt(eta);
            y_x_noisefree = y_x_noisefree + sum(sqrt(p).*abs(h).*s/sqrt(eta));
            %HENRIK STATIC
            [p_h, eta_h] = henrik_power(P_max, h, h_sort, sigma_w, T);
            p = p_h;
            eta = eta_h;
            %Calculate receive signal
            y_h = y_h + sum(sqrt(p).*abs(h).*s/sqrt(eta)) + w/sqrt(eta);
            %HENRIK DYNAMIC
            [p_hd, eta_hd] = henrik_power_dynamic(P_max, sigma_w, T, t, h_sort, old_p, old_h, old_eta);
            h = old_h(:,t);
            old_p(:,t) = p_hd;
            old_eta(t) = eta_hd;
            noise = noise + w/sqrt(eta_hd);
            p = p_hd;
            eta = eta_hd;
            %Calculate error
            y_hd = y_hd + sum(sqrt(p).*abs(h).*s/sqrt(eta)) + w/sqrt(eta);
            y_hd_noisefree = y_hd_noisefree + sum(sqrt(p).*abs(h).*s/sqrt(eta));
        end
        av_noise = av_noise + (noise/T)^2;
        true_sum = sum(s);
        x_error = real(true_sum-y_x/T)^2;
        %x_error = real(true_sum-y_x)^2;
        x_error_noisefree = real(true_sum-y_x_noisefree/T)^2;
        h_error = real(true_sum-y_h/T)^2;
        hd_error = real(true_sum-y_hd/T)^2;
        hd_error_noisefree = real(true_sum-y_hd_noisefree/T)^2;
        av_x_error = av_x_error + x_error;
        av_x_error_noisefree = av_x_error_noisefree + x_error_noisefree;
        av_h_error = av_h_error + h_error;
        av_hd_error = av_hd_error + hd_error;
        av_hd_error_noisefree = av_hd_error_noisefree + hd_error_noisefree;
    end
    x_errors(b) = av_x_error/num_av;
    x_errors_noisefree(b) = av_x_error_noisefree/num_av;
    h_errors(b) = av_h_error/num_av;
    hd_errors(b) = av_hd_error/num_av;
    hd_errors_noisefree(b) = av_hd_error_noisefree/num_av;
    noises(b) = av_noise/num_av;
    b
end
%% Plot comparison
x = 1:num_tests;
figure;
hold on;
plot(x, x_errors);
plot(x, h_errors);
%plot(x, hd_errors);
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
legend("Retransmission-unaware", "Retransmission-aware", 'Location', 'northwest');
xlabel("\sigma_w")
ylabel("Estimation error")
%% Plot misalignment/noise
x = 1:num_tests;
figure;
hold on;
plot(x, x_errors);
plot(x, x_errors_noisefree);
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
legend("xiaowen", "xiaowen w/o noise", 'Location', 'northwest');
xlabel("\sigma_w")
ylabel("Estimation error")
%% Plot misalignment/noise
x = 1:num_tests;
figure;
hold on;
plot(x, hd_errors);
plot(x, hd_errors_noisefree);
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
legend("henrik", "henrik w/o noise", 'Location', 'northwest');
xlabel("\sigma_w")
ylabel("Estimation error")