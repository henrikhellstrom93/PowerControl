function [b_ideal, eta_ideal] = henrik_ideal(P_max, H, s, sigma_w)
    n = length(P_max);
    b_bcd = min(P_max)*ones(n,1);
    eta_bcd = 1;
    num_it = 500;

    prev_obj = 10000;
    for d = 1:num_it
        if mod(d,10) == 0
            d
        end
        cvx_begin quiet
            variable b_bcd(n) complex
            minimize( (s-s.*b_bcd.*diag(H)*eta_bcd).'*conj(s-s.*b_bcd.*diag(H)*eta_bcd) + sigma_w^2*eta_bcd^2 )
            subject to
                abs(s).*abs(b_bcd) <= sqrt(P_max)
        cvx_end
        cvx_begin quiet
            variable eta_bcd
            minimize( (s.'*s) - 2*real((s.*b_bcd).'*(s.*diag(H)))*eta_bcd + real( ((s.*b_bcd).*conj(b_bcd)).'*((s.*diag(H)).*conj(diag(H))) )*eta_bcd^2 + sigma_w^2*eta_bcd^2 )
            subject to
                eta_bcd >= 0
        cvx_end
        curr_obj = (s-s.*b_bcd.*diag(H)*eta_bcd).'*conj(s-s.*b_bcd.*diag(H)*eta_bcd) + sigma_w^2*eta_bcd^2;
        if prev_obj/curr_obj < 1.0001
            break
        else
            prev_obj = curr_obj;
        end
    end
    b_ideal = b_bcd;
    eta_ideal = 1/eta_bcd^2;
end