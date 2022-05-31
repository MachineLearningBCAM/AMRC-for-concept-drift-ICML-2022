function [mistakes_rate, mistakes_idx, R_Ut] = AMRC(X, Y, k, W, N, K, deterministic, unidimensional, feature_mapping, varargin)
%{
  Input
  -----

      The name of dataset file

      k: Order

      W: Number of past labels
    
      N: Number of subgradients in the learning stage

      K: Number of iterations in the learning stage

      Deterministic: "True" for Deterministic AMRC and "False" for AMRC

      Unidimensional: "True" for Unidimensional AMRC and "False" for AMRC

      feature mapping

  Output
  ------

      Mistakes rate

      Mistakes indices

      R(U_t)

%}

% Length of the instance vectors
d = length(X(1,:));
% number of instances
T = length(X(:,1));
% number of classes
n_classes = length(unique(Y));

% Calculate the length m of the feature vector
if strcmp(feature_mapping,'linear')
    feature_parameters = [];
    m = n_classes*(d);
elseif strcmp(feature_mapping,'RFF')
    feature_parameters{1} = varargin{1};% random Fourier components D
    feature_parameters{2} = varargin{2}*randn(d, feature_parameters{1});
    m = n_classes*2*feature_parameters{1};
end

% Tracking uncertainty sets initializations
lambda = zeros(m, 1);
tau = zeros(m, 1);

% Learning AMRCs initializations
F = [];
h = [];
mu = zeros(m, 1);
mistakes_idx = [];
w = zeros(m, 1);
w0 = zeros(m, 1);

% Initialize mistakes counter
mistakes_idx = [];
mistakes = 0;

% Initialize mean vector estimate
[Ht, e1, eta, Sigma, eta0, Sigma0, epsilon, Q, R] = initialize_tracking(m, k);

for t = 1:T-1
    % New instance-label pair
    x = X(t, :)';% train instance vector
    y = Y(t);% train label
    
    % Estimating probabilities
    for i = 1:n_classes
        if t < W
            p(i, t) = length(find(Y(1:t) == i-1))/t;
            s(i, t) = std(p(i, 1:t));
        else
            p(i, t) = length(find(Y(t-W+1:t) == i-1))/W;
            s(i, t) = std(p(i, t-W+1:t));
        end
    end
    
    % Feature vector
    feature = feature_vector(x, y, n_classes, feature_mapping, feature_parameters);
    
    % Tracking uncertainty sets: update mean vector estimate and confidence vector
    [tau, lambda, eta, Sigma, eta0, Sigma0, epsilon, Q, R] = tracking(feature, y, k, Ht, eta, Sigma, eta0, Sigma0, epsilon, Q, R, e1, p(:, end), s(:, end), unidimensional);
    
    % Learning step, unipdate classifier parameter and obtain upper bound
    [mu, F, h, R_Ut(t), varphi,  w, w0] = learning(x, N, n_classes, mu, tau, lambda, F, h, w, w0, K, feature_mapping, feature_parameters);
    
    %  New  instance
    x_test = X(t+1, :)';% test instance
    
    % Prediction step, predict label for the new instance
    [hat_y,~] = predict_label(x_test, mu, n_classes, varphi, deterministic, feature_mapping, feature_parameters);
    
    % Receive the true label
    y_test = Y(t+1);% test label
    
    % Mistakes count
    if hat_y ~= y_test % Classification error
        mistakes_idx(t) = 1;
        mistakes = mistakes + 1;
    else
        mistakes_idx(t) = 0;
    end
end
mistakes_rate = mistakes/(T-1);
end
