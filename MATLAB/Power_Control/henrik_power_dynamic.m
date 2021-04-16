%ASSUMES P_max and h have been sorted by quality indicator
function [p_h, eta_h] = henrik_power_dynamic(P_max, sigma_w, T, t, h_sort, old_p, old_h, old_eta)
    K = length(P_max);
    
    %Find eta by taking the smallest eta_tilde
    eta_tilde = zeros(K,1);
    for k = 1:K
        sum1 = 0;
        sum2 = 0;
        for i = 1:k
            sum1 = sum1 + P_max(i)*abs(h_sort(i))^2;
            sum2 = sum2 + sqrt(P_max(i))*abs(h_sort(i));
        end
        eta_tilde(k) = ((sigma_w^2+1*sum1)/1/sum2)^2;
    end
    eta_h = min(eta_tilde);
    
    h = old_h(:,t);
    %Find p by channel inversion or max power
    p_h = zeros(K,1);
    for k = 1:K
        max_power = sqrt(P_max(k));
        sum = 0;
        for i = 1:t-1
            eta = old_eta(i);
            p = old_p(k,i);
            sum = sum + sqrt(eta_h)*sqrt(p)*abs(old_h(k,i))/sqrt(eta)/abs(h(k));
        end
        optimal_power = T*sqrt(eta_h)/abs(h(k))-sum;
        %There are some problems with rounding errors that I'm taking care
        %of here
        if optimal_power < 0.0001
            optimal_power = 0;
        end
        sqrt_p = min(max_power, optimal_power);
        p_h(k) = sqrt_p^2;
    end
end