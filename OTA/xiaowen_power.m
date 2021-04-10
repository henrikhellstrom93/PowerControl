%ASSUMES P_max and h have been sorted by quality indicator
function [p_x, eta_x] = xiaowen_power(P_max, h, h_sort, sigma_w)
    n = length(P_max);
    
    %Find eta by taking the smallest eta_tilde
    eta_tilde = zeros(n,1);
    for k = 1:n
        sum1 = 0;
        sum2 = 0;
        for i = 1:k
            sum1 = sum1 + P_max(i)*abs(h_sort(i))^2;
            sum2 = sum2 + sqrt(P_max(i))*abs(h_sort(i));
        end
        eta_tilde(k) = ((sigma_w^2+sum1)/sum2)^2;
    end
    eta_x = min(eta_tilde);
    
    %Find p by channel inversion or max power
    p_x = zeros(n,1);
    for i = 1:n
        if P_max(i) > eta_x/abs(h(i))^2
            %Possible to invert channel
            p_x(i) = eta_x/abs(h(i))^2;
        else
            %Maximum power used
            p_x(i) = P_max(i);
        end
    end
end