import torch
import torch.nn as nn
# Example of a simple linear layer in PyTorch

# Define a simple linear layer with one input feature and one output feature
m = nn.Linear(1, 1)

# Generate a random input tensor with values between 0 and 1 with shape (batch_size, num_features)
X = torch.rand(5, 1)  # Example with batch size of 5 and 1 feature
print(X)

# Generate an output tensor with the same shape as X with a slope of 2 and a y-intercept of 1

# Train the model
Y = 2 * X + 1
# Forward pass through the linear layer
Y_pred = m(X)
# Calculate the loss using Mean Squared Error
loss_fn = nn.MSELoss()
loss = loss_fn(Y_pred, Y)
# Print the loss
print("Loss:", loss.item())
# Backward pass to compute gradients
loss.backward()
# Print the gradients of the weights and bias
print("Gradient of weights:", m.weight.grad)
print("Gradient of bias:", m.bias.grad)
# Update the weights and bias using gradient descent
learning_rate = 0.01
with torch.no_grad():
    m.weight -= learning_rate * m.weight.grad
    m.bias -= learning_rate * m.bias.grad
# Reset gradients to zero for the next iteration
m.weight.grad.zero_()
m.bias.grad.zero_()
# Forward pass again to see the updated predictions
Y_pred_updated = m(X)
# Print the updated predictions
print("Updated Predictions:", Y_pred_updated)
# Print the updated weights and bias
print("Updated Weights:", m.weight)
print("Updated Bias:", m.bias)
# Print the final loss after the update
final_loss = loss_fn(Y_pred_updated, Y)
print("Final Loss:", final_loss.item())
# Print the final predictions
print("Final Predictions:", Y_pred_updated)
# Print the final target values
print("Final Target Values:", Y)
