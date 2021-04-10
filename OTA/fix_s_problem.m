clc
clear

num_tests = 20;
num_av = 1000;
noise_factor = 0.1;
heuristic_objs = zeros(num_tests, 1);
ideal_objs = zeros(num_tests, 1);
x_objs = zeros(num_tests, 1);
x_errors = zeros(num_tests, 1);
henrik_errors = zeros(num_tests, 1);
zero_errors = zeros(num_tests, 1);
norm_errors = zeros(num_tests, 1);
trunc_objs = zeros(num_tests, 1);
trunc_pos_objs = zeros(num_tests, 1);
trunc_errors = zeros(num_tests, 1);
trunc_pos_errors = zeros(num_tests, 1);
org_errors = zeros(num_tests, 1);
eta_list = zeros(num_tests, 1);
alt_eta_list = zeros(num_tests, 1);
x_snrs = zeros(num_tests, 1);
norm_snrs = zeros(num_tests, 1);
cvx_solver = false;
mse_s = true;

for a = 1:num_tests
    a
    av_x_obj = 0;
    av_x_error = 0;
    av_x_snr = 0;
    av_henrik_error = 0;
    av_zero_error = 0;
    av_norm_snr = 0;
    av_norm_error = 0;
    av_trunc_obj = 0;
    av_trunc_error = 0;
    av_trunc_pos_obj = 0;
    av_trunc_pos_error = 0;
    av_org_error = 0;
    av_heuristic_obj = 0;
    av_ideal_obj = 0;
    av_removal_obj = 0;
    for c = 1:num_av
        %Number of devices
        n=20;
        %Normally distributed data
        mu_s = 1;
        sigma_s = 1;
        s = sigma_s*randn(n,1) + mu_s;
        %Uniformly distributed data
        s_uni = 0.5*s;
        %que
        s_pos = abs(s);
        %Rayleigh fading
        H = diag( normrnd(0,1/2,n,1) + 1j*normrnd(0,1/2,n,1) );
        %Eta initialized to 1
        eta = 1;
        eta_bcd = 1;
        %Unit variance noise
        sigma_w = noise_factor*a;
        %Noise
        w = normrnd(0,sigma_w);
        %Power constraint
        P_max = 10*ones(n,1);
        %P_max = 0.3*rand(n,1)+0.1*(a+5);
        %P_max = 0.3*rand(n,1)+0.1*(15);
        %b initialized to smallest b_max
        b = min(P_max)*ones(n,1);
        b_bcd = min(P_max)*ones(n,1);
        %Sort P_max and H by quality indicator
        [quality_indicator, sort_index] = sort(abs(H).^2*P_max);
        P_max_sort = zeros(n,1);
        H_sort = zeros(n,n);
        for i = 1:n
            P_max_sort(i) = P_max(sort_index(i));
            H_sort(i,i) = H(sort_index(i), sort_index(i));
        end
        P_max = P_max_sort;
        H = H_sort;
        
        %Xiaowen solution
        [b_x, eta_x] = xiaowen(P_max, H, sigma_w);
        true_sum = sum(s);
        mu_n = 0;
        sigma_n = 1;
        s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
        y = sum(b_x.*diag(H).*s_norm/sqrt(eta_x)) + w/sqrt(eta_x);
        est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, n);
        x_error = real(true_sum-est_sum)^2;
        av_x_error = av_x_error + x_error;
        
        %Henrik norm solution
        [b_norm, eta_norm, mu_n, sigma_n] = henrik_norm(P_max, H, sigma_w, mu_s, sigma_s);
        s_norm = normalize(s, mu_s, sigma_s, mu_n, sigma_n);
        true_sum = sum(s);
        y = sum(b_norm.*diag(H).*s_norm/sqrt(eta_norm)) + w/sqrt(eta_norm);
        est_sum = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, n);
        henrik_error = real(true_sum-est_sum)^2;
        av_henrik_error = av_henrik_error + henrik_error;
        
        %Xiaowen truncated solution
        [b_trunc] = xiaowen_trunc(P_max, H, s, b_x);
        eta_trunc = eta_x;
        
        
        %Xiaowen truncated positive s
        [b_trunc_pos] = xiaowen_trunc(P_max, H, s_pos, b_x);
        eta_trunc_pos = eta_x;
        true_sum = sum(s_pos);
        est_sum = real(((b_trunc_pos.*s_pos).'*diag(H) + w)/sqrt(eta_trunc_pos));
        est_sum_noiseless = real((b_trunc_pos.*s_pos).'*diag(H)/sqrt(eta_trunc_pos));
        error = abs(true_sum - est_sum);
        av_trunc_pos_error = av_trunc_pos_error + error;
        
        %Henrik heuristic
        [b_heuristic] = henrik_heuristic(P_max, H, s, eta_x);
        eta_heuristic = eta_x;
        
        %Henrik ideal
        if cvx_solver == true
            [b_ideal, eta_ideal] = henrik_ideal(P_max, H, s, sigma_w);
        else
            [b_ideal, eta_ideal] = henrik_ideal2(P_max, H, s, sigma_w);
        end
        
        %Full inversion
        [b_inv, eta_inv] = inversion(P_max, H, s);

        %Relative powers for plotting
        p_x_relative = abs(b_x)./sqrt(P_max);
        p_henrik_relative = abs(b_norm)./(sqrt(P_max)*sigma_s/sqrt(mu_s^2+sigma_s^2));
        for i = 1:n
            if p_henrik_relative > 1
                disp("I AM CHEATING!")
                return
            end
        end
        p_trunc_relative = abs(s).*abs(b_trunc)./sqrt(P_max);
        p_heuristic_relative = abs(s).*abs(b_heuristic)./sqrt(P_max);
        p_ideal_relative = abs(s).*abs(b_ideal)./sqrt(P_max);
        
        %All objective functions
        if mse_s == true
            x_obj = (s-s.*b_x.*diag(H)/sqrt(eta_x)).'*conj(s-s.*b_x.*diag(H)/sqrt(eta_x)) + sigma_w^2/eta_x;
            trunc_obj = (s-s.*b_trunc.*diag(H)/sqrt(eta_trunc)).'*conj(s-s.*b_trunc.*diag(H)/sqrt(eta_trunc)) + sigma_w^2/eta_trunc;
            trunc_pos_obj = (s_pos-s_pos.*b_trunc_pos.*diag(H)/sqrt(eta_trunc)).'*conj(s_pos-s_pos.*b_trunc_pos.*diag(H)/sqrt(eta_trunc)) + sigma_w^2/eta_trunc;
            heuristic_obj = (s-s.*b_heuristic.*diag(H)/sqrt(eta_heuristic)).'*conj(s-s.*b_heuristic.*diag(H)/sqrt(eta_heuristic)) + sigma_w^2/eta_heuristic;
            ideal_obj = (s-s.*b_ideal.*diag(H)/sqrt(eta_ideal)).'*conj(s-s.*b_ideal.*diag(H)/sqrt(eta_ideal)) + sigma_w^2/eta_ideal;
        else
            x_obj = n - 2*real(ones(1,n)*H*b_x)/sqrt(eta_x) + real((H*b_x)'*H*b_x)/eta_x + sigma_w^2/eta_x;
            trunc_obj = n - 2*real(ones(1,n)*H*b_trunc)/sqrt(eta_trunc) + real((H*b_trunc)'*H*b_trunc)/eta_trunc + sigma_w^2/eta_trunc;
            trunc_pos_obj = n - 2*real(ones(1,n)*H*b_trunc)/sqrt(eta_trunc) + real((H*b_trunc)'*H*b_trunc)/eta_trunc + sigma_w^2/eta_trunc;
            heuristic_obj = n - 2*real(ones(1,n)*H*b_heuristic)/sqrt(eta_heuristic) + real((H*b_heuristic)'*H*b_heuristic)/eta_heuristic + sigma_w^2/eta_heuristic;
            ideal_obj = n - 2*real(ones(1,n)*H*b_ideal)/sqrt(eta_ideal) + real((H*b_ideal)'*H*b_ideal)/eta_ideal + sigma_w^2/eta_ideal;
        end
        av_x_obj = av_x_obj + x_obj;
        av_trunc_obj = av_trunc_obj + trunc_obj;
        av_trunc_pos_obj = av_trunc_pos_obj + trunc_pos_obj;
        av_heuristic_obj = av_heuristic_obj + heuristic_obj;
        av_ideal_obj = av_ideal_obj + ideal_obj;
        
        if c == num_av
            x_objs(a) = av_x_obj/num_av;
            x_errors(a) = av_x_error/num_av;
            henrik_errors(a) = av_henrik_error/num_av;
            zero_errors(a) = av_zero_error/num_av;
            norm_errors(a) = av_norm_error/num_av;
            trunc_objs(a) = av_trunc_obj/num_av;
            trunc_errors(a) = av_trunc_error/num_av;
            trunc_pos_objs(a) = av_trunc_pos_obj/num_av;
            trunc_pos_errors(a) = av_trunc_pos_error/num_av;
            heuristic_objs(a) = av_heuristic_obj/num_av;
            ideal_objs(a) = av_ideal_obj/num_av;
            org_errors(a) = av_org_error/num_av;
            norm_errors(a) = av_norm_error/num_av;
            x_snrs(a) = av_x_snr/num_av;
            norm_snrs(a) = av_norm_snr/num_av;
        end
    end
    
end    
% Plot errors
x = 1:num_tests;
figure;
hold on;
plot(x, x_errors);
plot(x, henrik_errors);
legend("xiaowen", "henrik", 'Location', 'northwest');
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
xlabel("\sigma_w")
ylabel("Estimation error")

%% Plot normalization comparison
x = 1:num_tests;
figure;
hold on;
plot(x, x_errors);
plot(x, zero_errors);
legend("xiaowen", "sum=0", 'Location', 'northwest');
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
xlabel("\sigma_w")
ylabel("Estimation error")

%% Plot SNR comparison
x = 1:num_tests;
figure;
hold on;
plot(x, x_snrs);
plot(x, norm_snrs);
legend("zero mean", "Unit mean", 'Location', 'northwest');
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
xlabel("\sigma_w")
ylabel("SNR")
%% Plot positive comparison
x = 1:num_tests;
figure;
hold on;
plot(x, trunc_errors);
plot(x, trunc_pos_errors);
legend("normalized", "positive normalized", 'Location', 'northwest');
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
xlabel("\sigma_w")

%% Plot heuristic ideal comparison
x = 1:num_tests;
figure;
hold on;
plot(x, heuristic_objs);
plot(x, ideal_objs);
legend("heuristic", "ideal", 'Location', 'northwest');
xticks(2:2:num_tests)
xticklabels(0.1*(2:2:num_tests))
xlabel("\sigma_w")
%% Plot bar diagram xiaowen
figure;
bar(1:n,p_x_relative)
ylim([0 1.2])
xlabel("Device index")
ylabel("Relative transmission power")
%% Plot bar diagram henrik
figure;
bar(1:n,p_henrik_relative)
ylim([0 1.2])
xlabel("Device index")
ylabel("Relative transmission power")
%% Plot bar diagram xiaowen noncheat
figure;
bar(1:n,p_trunc_relative)
ylim([0 3])
xlabel("Device index")
ylabel("Relative transmission power")
%% Plot bar diagram henrik
figure;
bar(1:n,p_heuristic_relative)
ylim([0 3])
xlabel("Device index")
ylabel("s_k^2|h_k|^2/P_k")
%% Plot results
x = 1:num_tests;
figure;
hold on;
plot(x, trunc_objs);
if cvx_solver == false
    %plot(x, x_objs)
else
    plot(x, x_objs)
    plot(x, ideal_objs);
end
plot(x, heuristic_objs);
xticks(2:2:num_tests)
xticklabels(noise_factor*(2:2:num_tests))
if cvx_solver == false
    legend("truncated statistical", "self-knowledge", 'Location', 'northwest');
else
    legend("truncated statistical", "statistical", "ideal self-knowledge", "heuristic self-knowledge", 'Location', 'northwest');
end
xlabel("\sigma_w")
ylabel("MSE")
%% Plot mean result
x = 1:3;
figure;
hold on;
%bar(x, [mean(henrik_objs), mean(xiaowen_noncheat_objs)])
bar(x, [mean(heuristic_objs), mean(trunc_objs), mean(x_objs)])
%% Plot s factors
henrik_factor = real(H*b_heuristic/sqrt(eta_heuristic));
xiaowen_factor = real(H*b_x/sqrt(eta_x));
removal_factor = real(H*b_removal/sqrt(eta_removal));
xiaowen_noncheat_factor = real(H*b_trunc/sqrt(eta_trunc));
x = 1:20;
figure;
hold on;
%plot(x, henrik_factor)
plot(x, xiaowen_factor)
plot(x, removal_factor)
%plot(x, xiaowen_noncheat_factor)