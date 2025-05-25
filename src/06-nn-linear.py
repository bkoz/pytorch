import torch
import torch.nn as nn


model = nn.Linear(1, 1, bias=True)  # Simple linear model with one input and one output
input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Input should be a tensor of shape (10,)
target = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # Target should be a tensor of shape (10,)
print(f'{target.size()=}')
print(f'{target=}')

print(f'{input.size()=}')
print(f'{input=}')

# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)        
for epoch in range(100):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    output = model(input.float().view(-1, 1))  # Reshape input to (10, 1)
    loss = criterion(output, target.float().view(-1, 1))  # Reshape target to (10, 1)
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
# Test the model
model.eval()
with torch.no_grad():
    test_input = torch.tensor([11, 12, 13, 14, 15])  # Test input
    test_output = model(test_input.float().view(-1, 1))  # Reshape test input to (5, 1)
    print(f'Test Input: {test_input}')
    print(f'Test Output: {test_output.squeeze()}')  # Squeeze to remove extra dimension

# Print the model parameters
print(f'Model Parameters: {list(model.parameters())}')
# Print the learned weights and bias
print(f'Learned Weights: {model.weight.data}')
print(f'Learned Bias: {model.bias.data}')

# Print the model in the y = mx + b format
print(f'Model: y = {model.weight.data.item()} * x + {model.bias.data.item()}')
