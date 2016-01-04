function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

matrix_of_e = ones(size(z)) * e;
denom = 1 + matrix_of_e .** ((-1) * z);
g = ones(size(z)) ./ denom;



% =============================================================

end
