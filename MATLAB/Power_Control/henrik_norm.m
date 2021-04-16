function [b_norm, eta_norm, mu_n, sigma_n] = henrik_norm(P_max, H, sigma_w, mu_s, sigma_s)
    n = length(P_max);
    sigma_n = 1;
    mu_n = mu_s/sigma_s;
    
    %Find eta by taking the smallest eta_tilde
    eta_tilde = zeros(n,1);
    for k = 1:n
        sum1 = 0;
        sum2 = 0;
        for i = 1:k
            sum1 = sum1 + P_max(i)*abs(H(i,i))^2;
            sum2 = sum2 + sqrt(P_max(i))*abs(H(i,i));
        end
        eta_tilde(k) = sigma_s^2/(mu_s^2+sigma_s^2)*((sigma_w^2+sum1)/sum2)^2;
    end
    eta_norm = min(eta_tilde);
    
    %Find b by channel inversion or max power
    b_norm = zeros(n,1);
    for i = 1:n
        if P_max(i)*sigma_s^2/(mu_s^2+sigma_s^2) > eta_norm/abs(H(i,i))^2
            %Channel can be inverted
            b_norm(i) = conj(H(i,i))*sqrt(eta_norm)/abs(H(i,i))^2;
        else
            %Channel cannot be inverted
            b_norm(i) = conj(H(i,i))*sqrt(P_max(i))/abs(H(i,i))*sigma_s/sqrt(mu_s^2+sigma_s^2);
        end
    end
end