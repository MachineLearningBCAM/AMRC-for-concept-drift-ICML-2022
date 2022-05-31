function phi=feature_vector(x,y, n_classes, feature_mapping, feature_parameters)
%{
   Feature mappings

   This function obtains feature vectors

   Input
   -----

   x: new instance

   y: new label

   n_classes: number of classes

   feature_mapping: 'linear' or 'kernel'

   feature_parameters:
       if feature_mapping == 'linear': feature_parameters = []
       if feature_mapping == 'kernel': feature_parameters = [D, u] where
           D = number of random Fourier components
           u = random Fourier components


   Output
   ------

   phi: feature vector

%}

if strcmp(feature_mapping,'linear')
    x_phi = [x];
elseif strcmp(feature_mapping,'kernel')
    D_kernel = feature_parameters{1};
    u = feature_parameters{2};
    x_phi = [cos(u'*x); sin(u'*x)];
end
    e = zeros(n_classes, 1);
    e(y+1) = 1;
    phi = kron(e, x_phi);
end
