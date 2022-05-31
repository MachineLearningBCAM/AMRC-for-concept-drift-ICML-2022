function [Ht, e1, eta, Sigma, eta0, Sigma0, epsilon, Q, R] = initialize_tracking(m, k)
%{
   Initialize tracking stage

   This function defines matrices and vectors that are used to update mean vector estimates and confidence vectors.

   Input
   -----

   m: length of mean vector estimate

   k: order

   Output
   ------

   Ht: transition matrix

   e1: vector with 1 in the first component and 0 in the remainning components

   eta: state vectors

   Sigma: mean squared error matrices

   eta0, Sigma0, epsilon: parameters required to obtain variances of noise processes

   Q, R: variances of noise processes

%}
 
e1 = zeros(1, k+1);
e1(1) = 1;
deltat = 1;
variance_init = 0.001;
Ht = eye(k+1);
for i = 1:k
    for j = i+1:k+1
        Ht(i, j) = deltat^(j-i)/(factorial(j-i));
    end
end
eta0 = zeros(k+1, m);
for i = 1:m
    Sigma0(:, :, i) = eye(k+1);
    Q(:, :, i) = variance_init*eye(k+1, k+1);
    R(i) = variance_init;
    epsilon(i) = 0 - e1*eta0(:, i);
    eta(:, i) = Ht*eta0(:, i);
    Sigma(:, :, i) = Ht*Sigma0(:, :, i)*Ht'+Q(:, :, i);
end
end