function [y, h]=predict_label(x, mu, n_classes, varphi, deterministic, feature_mapping, feature_parameters)
%{

   Predict

   This function assigns labels to instances

   Input
   -----

   x: instance

   mu: classifier parameter

   n_classes: number of classes

   varphi: varphi function obtained at learning

   deterministic: "True" for deterministic AMRC and "False" for AMRC

   feature_mapping: 'linear' or 'kernel'

   feature_parameters:
       if feature_mapping == 'linear': feature_parameters = []
       if feature_mapping == 'kernel': feature_parameters = [D, u] where
           D = number of random Fourier components
           u = random Fourier components

   Output
   ------

   y_pred: predicted label

%}

for j=1:n_classes
    M(j,:)=feature_vector(x, j-1, n_classes, feature_mapping, feature_parameters)';
end
for i=1:n_classes
    c(i) = max([(M(i, :)*mu-varphi), 0]);
end
cx = sum(c);
for i=1:n_classes
    if cx == 0
        h(i)=1/n_classes;
    else
        h(i)=c(i)/cx;
    end
end
if deterministic == 'True'
    y = find(h == max(h));
    y = y(1)-1;
elseif deterministic == 'False'
    y=find(mnrnd(1,h)==1)-1;
else
    print('Error')
end
end