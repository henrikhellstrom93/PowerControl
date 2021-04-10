function [b_heuristic] = henrik_heuristic(P_max, H, s, eta_x)
    n = length(P_max);
    
    %Find b by channel inversion or max power
    b_heuristic = zeros(n,1);
    for i = 1:n
        if P_max(i) > abs(s(i))^2*eta_x/abs(H(i,i))^2
            b_heuristic(i) = conj(H(i,i))*sqrt(eta_x)/abs(H(i,i))^2;
        else
            b_heuristic(i) = conj(H(i,i))/abs(H(i,i))*sqrt(P_max(i))/abs(s(i));
        end
    end
end