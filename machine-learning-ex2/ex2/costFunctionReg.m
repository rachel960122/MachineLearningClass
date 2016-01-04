function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;

denom = 1 + exp(-z);
hypo = ones(size(z)) ./ denom;

pos = y .* log(hypo);
neg = (1 - y) .* log(1 - hypo);
reg_theta = theta(2:size(theta, 1), :);
reg = (lambda / (2 * m)) * sum(reg_theta .^ 2);
J = (- 1 / m) * sum(pos + neg) + reg;

d_hypo = (1 / m) * (sum((hypo - y) .* X))';
d_reg = (lambda / m) * theta;
grad_zero = d_hypo(1, :);
grad_rest = d_hypo(2:size(theta, 1), :) + d_reg(2:size(theta, 1), :);
grad = vertcat(grad_zero, grad_rest);



% =============================================================

end
