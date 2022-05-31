clear all;
close all;
%{
Input
-----

    The name of dataset file

    Order

    lambda0

    feature mapping

    Deterministic: "True" for Deterministic AMRC and "False" for AMRC

    Unidimensional: "True" for Unidimensional AMRC and "False" for AMRC

Output
------

    Mistakes rate

    Mistakes indexes

    R_Ut
%}

filename = 'usenet2'; % 'name_dataset';
temp = [filename,'.mat'];
cd ..
load(temp)
cd Matlab_code

% Normalize data
X = rescale(X);

% Choose the Order
order = 1;
W = 200;

% Nesterov's-SG
N = 100;
K = 2000;

% Deterministic AMRC or AMRC
deterministic = "True";

% Unidimensional AMRC or AMRC
unidimensional = "False";
if unidimensional  ==  "False"
    order = order;
elseif unidimensional == "True"
    order = 0;
else
    print('Error');
end

% Choose a feature mapping: 'linear' or 'RFF'
feature_mapping = 'linear';
[mistakes_rate, mistakes_idx, R_Ut] = AMRC(X, Y, order, W, N, K, deterministic, unidimensional, feature_mapping);
% If feature_mapping == 'linear' then, there is not optional arguments.
% However, if feature_mapping == 'RFF' then, varargins are
    % feature_mapping = 'RFF';
    % D = 200 (number of random Fourier components)
    % gamma (scaling factor of RFF features)
% [mistakes_rate, mistakes_idx, R_Ut] = AMRC(X, Y, order, W, N, K, feature_mapping, D, gamma);

disp(['AMRC for order ', num2str(order), ' has a mistakes rate ', num2str(mistakes_rate*100), '%', ' in ', filename, ' dataset using ', feature_mapping, ' feature mapping'])

temp = "%s_results";
filename = sprintf(temp,filename);
save(filename)
