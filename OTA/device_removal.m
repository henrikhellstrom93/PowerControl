function [b_removal, eta_removal] = device_removal(P_max, H, sigma_w, s, threshold)
    n = length(P_max);
    
    %If device signal is below threshold, remove from consideration
    remove_index = [];
    for i = 1:n
        if abs(s(i)) < threshold
            remove_index = [remove_index i]; %It's a small vector, no need to preallocate
        end
    end
    P_max(remove_index) = [];
    h = diag(H);
    h(remove_index) = [];
    H = diag(h);
    
    [b_removal, eta_removal] = xiaowen(P_max, H, sigma_w);
    for i=remove_index
        if i == 1
            b_removal = [0; b_removal];
        else
            b_removal = [b_removal(1:remove_index-1); 0; b_removal(remove_index:end)];
        end
    end
end