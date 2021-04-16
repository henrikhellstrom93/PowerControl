function [s_normalized] = normalize(s, mu_s, sigma_s, mu_n, sigma_n)
    s_normalized = (s-mu_s)/sigma_s*sigma_n + mu_n;
end