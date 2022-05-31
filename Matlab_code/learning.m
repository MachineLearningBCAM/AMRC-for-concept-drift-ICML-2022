function [mu, F, h, R_Ut, varphi, w, w0] = learning(x, N, n_classes, mu, tau, lmb, F, h, w, w0, K, feature_mapping, feature_parameters)
%{
Learning

    This function efficientle learns classifier parameters

    Input
    -----

    x: instance

    N: number of subgradients

    n_classes: number of classes

    mu: classifier parameter

    tau: mean vector estimate

    lmb: confidence vector

    F: matrix that is used to obtain local approximations of function varphi

    h: vector that is used to obtain local approximations of function varphi

    w, w0: Nesterov's-SG parameters

    K: number of iterations Nesterov's-SG

    feature_mapping: 'linear' or 'RFF'

    feature_parameters:
        if feature_mapping == 'linear': feature_parameters = []
        if feature_mapping == 'RFF': feature_parameters = [D, u] where
            D = number of random Fourier components
            u = random Fourier components

    Output
    ------

    mu: updated classifier parameter

    F: updated matrix that is used to obtain local approximations of function varphi

    h: updated vector that is used to obtain local approximations of function varphi

    R_Ut: upper bounds

    varphi: function that is then used at prediction

    w, w0: Nesterov's-SG parameters
%}

theta = 1;
theta0 = 1;
d = length(x(:, 1));
muaux = mu;
R_Ut = 0;
M = [];
for j = 1:n_classes
    M(end+1,:)=feature_vector(x,j-1,n_classes, feature_mapping, feature_parameters)';
end
for j=1:n_classes
    aux=nchoosek(1:n_classes,j);
    for k=1:length(aux(:,1))
        idx=zeros(1,n_classes);
        idx(aux(k,:))=1;
        F(end+1, :) = (idx*M)./j;
        h(end+1, 1) = - 1/j;
    end
end
v = F*muaux + h;
[varphi, ~] = max(v');
    regularization = 0;
     for i = 1:length(lmb)
        regularization = regularization + (lmb(i)*abs(muaux(i)));
     end
R_Ut_best_value = 1 - tau*muaux + varphi + regularization;
F_count = zeros(length(F(:, 1)), 1);
for l = 1:K
    muaux = w + theta*((1/theta0) - 1)*(w-w0);
    v = F*muaux + h;
    [varphi, idx_mv] = max(v');
    fi = F(idx_mv, :);
    F_count(idx_mv) = F_count(idx_mv) + 1;
    regularization = 0;
     for i = 1:length(lmb)
        subgradient_regularization(i) = lmb(i)*sign(muaux(i));
        regularization = regularization + (lmb(i)*abs(muaux(i)));
     end
      g = - tau' + fi' + subgradient_regularization';
     theta0 = theta;
     theta = 2/(l+1);
     alpha = 1/((l+1)^(3/2));
     w0 = w;
     w = muaux - alpha*g;
    R_Ut = 1 - tau*muaux + varphi + regularization;
    if R_Ut < R_Ut_best_value
        R_Ut_best_value = R_Ut;
        mu = muaux;
    end
    end
v = F*w + h;
[varphi, ~] = max(v');
regularization = 0;
     for i = 1:length(lmb)
        regularization = regularization + (lmb(i)*abs(w(i)));
     end
R_Ut = 1 - tau*w + varphi + regularization;
if R_Ut < R_Ut_best_value
    R_Ut_best_value = R_Ut;
    mu = w;
end
if length(F(:, 1)) > N
    idx_F_count = find(F_count == 0);
    if length(idx_F_count) > length(F(:, 1)) - N
        h(idx_F_count(1:(length(F(:, 1)) - N)), :) = [];
        F(idx_F_count(1:(length(F(:, 1)) - N)), :) = [];
    else
        h(1:(length(idx_F_count) - length(F(:, 1)) + N), :) = [];
        F(1:(length(idx_F_count) - length(F(:, 1)) + N), :) = [];
        F(idx_F_count, :) = [];
        h(idx_F_count, :) = [];
    end
end
end
