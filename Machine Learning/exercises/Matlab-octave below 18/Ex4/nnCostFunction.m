function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_p = y;

X_p = [ones(m,1), X];
a_1 = sigmoid(X_p * Theta1'); %size : m * H1 (first hidden layer size)
a_1 = [ones((m),1), a_1];  
a_2 = sigmoid(a_1 * Theta2'); 

h_theta = a_2;  %size : m*k


for k =1:num_labels
  
     for i=1:m   
         
        if y_p(i) == k
            y_p(i) = 1;
            
        else
            y_p(i) = 0;
            
        end
        
         %J = J - 1/m * ( y_p(i) * log(h_theta(i,k)) ...
          %  +(1 - y_p(i)) * log(1 - h_theta(i,k)) );
     end
     
     J = J - 1/m * sum( y_p .* log(h_theta(:,k)) ...
            + (1 - y_p).* log(1 - h_theta(:,k)) );
        
        y_p = y;
            
end

%cost function with regularization:

theta1 = Theta1(:, 2:end);
theta2 = Theta2(:, 2:end);
J = J + (lambda/(2*m))*(sum(theta1.^2, 'all') + sum(theta2.^2, 'all'));





%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
yv=[1:num_labels] == y





            


% for t=1:m
%     
%     %STEP 1:
%     a_1 = X(t,:)'; 
%     a_1 = [1;a_1]; %size : (n+1)*1
%     
%     z_2 = Theta1 * a_1; 
%     a_2 = sigmoid(z_2); 
%     a_2 = [1;a_2];%size: (H+1) * 1
%     
%     z_3 = Theta2 *a_2;
%     a_3 = sigmoid(z_3); %size = num_labels * 1
%     
%     
%     %STEP 2:
%     
%     delta_3 = zeros(size(a_3));
%     idx = 0;
%     for k=1:num_labels
%         
%         if y_p(t) == k
%             
%             idx = 1;
%             
%         else
%             idx = 0 ;
%         end
%         
%         delta_3(k) = a_3(k) - idx;
%         
%         
%     end
%     
%     
%     %STEP 3:
%     %?2 equals the product of ?3 and ?2 (ignoring the ?2 bias units)
%     Theta2_p = Theta2(:,2:end); %we do not include the first column
%     Thedel = Theta2_p' * delta_3; %size = H*1
%     grad_z = sigmoidGradient(z_2); %size = H*1 
%     
%     
%     delta_2 = Thedel .* grad_z ; %size = H *1
%     
%     
%     %a_1 = a_1(2:end, :);
%     %a_2 = a_2(2:end, :);
%     
%     %STEP 4
%     % gradient matrices are the same size as Theta1 and Theta2
%     Delta_1 = zeros(hidden_layer_size, (input_layer_size + 1));
%     Delta_2 = zeros(num_labels, (hidden_layer_size + 1));
%     
%     Delta_1 = Delta_1 + delta_2 * a_1';
%     Delta_2 = Delta_2 + delta_3 * a_2';
%     
%     
% end
% 
% %STEP 5
% Theta1_grad = 1/m * Delta_1;
% Theta2_grad = 1/m * Delta_2;





% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
