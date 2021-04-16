function [b_inv, eta_inv] = inversion(P_max, H, s)
    n = length(P_max);
    
    %Find eta so that weakest device can invert
    eta_inv = 100000000000;
    for i = 1:n
        if eta_inv > abs(H(i,i))^2*P_max(i)/abs(s(i))^2
            eta_inv = abs(H(i,i))^2*P_max(i)/abs(s(i))^2;
        end
    end
    
    %Invert all channels
    b_inv = zeros(n,1);
    for i = 1:n
        b_inv(i) = sqrt(eta_inv)*conj(H(i,i))/abs(H(i,i))^2;
    end
end