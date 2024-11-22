function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%a column of ones is already added to X
% theta_p = [1; theta];
h_theta = X * theta;
theta_p = theta(2:end);

J = 1/(2*m) * sum((h_theta - y).^2, 'all') + lambda/(2*m)*sum(theta_p.^2,'all');


grad(1) = 1/m * sum(h_theta - y);
for j=2:size(theta,1)
grad(j) = 1/m * sum((h_theta - y).*X(:,j)) + lambda/m * theta(j);

end
















% =========================================================================

grad = grad(:);

end
