function [b_ideal, eta_ideal] = henrik_ideal2(P_max, H, s, sigma_w)
    n = length(P_max);
    
    %Find eta by taking the smallest eta_tilde
    eta_tilde = zeros(n,1);
    for k = 1:n
        sum1 = 0;
        sum2 = 0;
        for i = 1:k
            sum1 = sum1 + P_max(i)*abs(H(i,i))^2;
            sum2 = sum2 + sqrt(P_max(i))*abs(H(i,i))*abs(s(i));
        end
        eta_tilde(k) = ((sigma_w^2+sum1)/sum2)^2;
    end
    eta_ideal = min(eta_tilde);
    
    %Find b by channel inversion or max power
    b_ideal = zeros(n,1);
    for i = 1:n
        if P_max(i) > abs(s(i))^2*eta_ideal/abs(H(i,i))^2
            b_ideal(i) = conj(H(i,i))*sqrt(eta_ideal)/abs(H(i,i))^2;
        else
            b_ideal(i) = conj(H(i,i))/abs(H(i,i))*sqrt(P_max(i))/abs(s(i));
        end
    end
end