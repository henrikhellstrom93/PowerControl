function [b_trunc] = xiaowen_trunc(P_max, H, s, b_x)
    n = length(P_max);
    
    b_trunc = zeros(n,1);
    for i = 1:n
        if abs(b_x(i)) > sqrt(P_max(i))/abs(s(i))
            b_trunc(i) = conj(H(i,i))/abs(H(i,i))*sqrt(P_max(i))/abs(s(i));
        else
            b_trunc(i) = b_x(i);
        end
    end
end