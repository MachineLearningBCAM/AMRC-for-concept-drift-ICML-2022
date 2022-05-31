function [tau, lmb, eta, Sigma, eta0, Sigma0, epsilon, Q, R] = tracking(feature, y, k, Ht, eta, Sigma, eta0, Sigma0, epsilon, Q, R, e1, p, s, unidimensional)
%{
     Tracking uncertainty sets

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     feature: feature vector

     y: new label

     n_classes: number of classes

     eta: state vector estimate composed by mean vector estimate and its k derivatives

     Sigma: mean quadratic error matrix

     Ht: transition matrix

     D: diagonal matrix

     eta0, Sigma0, epsilon: parameters required to update variances of noise processes

     e1: vector with 1 in the first component and 0 in the remainning components

     Q, R: variances of random noises

     p, s: probability

     unidimensional: "True" for unidimensional AMRC and "False" for AMRC

     Output
     ------

     eta: updated mean vector estimate

     Sigma: updated mean quadratic error matrix

     tau: mean vector estimate

     lmb: confidence vector

     eta0, Sigma0, epsilon: parameters required to update variances of noise processes
     
     Q, R: variances of noise processes
%}

m = length(feature);
n_classes = length(p(:, 1));
d = m/n_classes;
alpha = 0.3;
if unidimensional == "True"
    for i = 1:m
        R(i) = alpha*R(i) + (1-alpha)*(epsilon(i)^2 + Sigma0(:, :, i));
        innovation(i) = mean(feature) - eta0(:, i);
        KK(i) = (Sigma0(:, :, i))*(Sigma0(:, :, i)+ R(i))^(-1);
    end
    K = mean(KK);
    for i = 1:m
        eta0(:, i) = eta(:, i) + K*innovation(i);
        Sigma(:, :, i) = (1 - K)*Sigma0(:, :, i);
        Q(i) = alpha*Q(i) + (1-0.3)*(innovation(i)^2)*(K*K');
        epsilon(i) = mean(feature) - e1*eta(:, i);
        eta0(:, i) = eta(:, i);
        Sigma0(:, :, i) = Sigma(:, :, i)+Q(i);
        tau(i) = (1/n_classes)*(eta0(:, i));%
        lmb_eta(i) = (sqrt(Sigma0(:, :, i)));
        lmb(i) = mean(lmb_eta(i));
    end
elseif unidimensional == "False"
    for i = 1:m
        if i > (y)*(d) && i < (y+1)*(d)+1
            innovation(i) = feature(i) - e1*eta(:, i);
            R(i) = alpha*R(i) + (1-alpha)*(epsilon(i)^2 + e1*Sigma(:, :, i)*e1');
            K = (Sigma(:, :, i)*e1')*(e1*Sigma(:, :, i)*e1' + R(i))^(-1);
            eta0(:, i) = eta(:, i) + K*innovation(i);
            Sigma0(:, :, i) = (eye(k+1) - K*e1)*Sigma(:, :, i);
            Q(:, :, i) = alpha*Q(:, :, i) + (1-alpha)*(innovation(i)^2)*(K*K');
            epsilon(i) = feature(i) - e1*eta0(:, i);
            eta(:, i) = Ht*eta0(:, i);
            Sigma(:, :, i) = Ht*Sigma0(:, :, i)*Ht'+Q(:, :, i);
            tau(i) = (p(y+1, end))*(e1*eta(:, i));%
            lmb_eta(i) = (sqrt(Sigma(1, 1, i)));
            lmb(i) =sqrt((lmb_eta(i)^2 + (e1*eta(:, i))^2)*(s(y+1, end)^2 + p(y+1, end)^2) - ((e1*eta(:, i))^2)*(p(y+1, end)^2));
        else
            eta(:, i) = Ht*eta0(:, i);
            Sigma(:, :, i) = Ht*Sigma0(:, :, i)*Ht'+Q(:, :, i);
            tau(i) = (p(floor((i-1)/d)+1, end))*(e1*eta(:, i));
            lmb_eta(i) = (sqrt(Sigma(1, 1, i)));
            lmb(i) = sqrt((lmb_eta(i)^2 + (e1*eta(:, i))^2)*(s(floor((i-1)/d)+1, end)^2 + (p(floor((i-1)/d)+1, end))^2) - ((e1*eta(:, i))^2)*(p(floor((i-1)/d)+1, end))^2);
        end
    end
else
    print('Error')
end
end