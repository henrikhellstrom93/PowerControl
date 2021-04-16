function [denormalized_sum] = denormalize(y, mu_s, sigma_s, mu_n, sigma_n, n)
    denormalized_sum = (y-n*mu_n)*sigma_s/sigma_n+n*mu_s;
end