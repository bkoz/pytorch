import numpy as np
import math
import matplotlib.pyplot as plt

num_points = 2000
# Create n_points of input and random output data.
x = np.linspace(-math.pi, math.pi, num_points)
# Perturb the y values with some noise.
y = 2 * x + np.random.randn(num_points) * 0.1 + 1

# plot these points
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
plt.grid(True)
plt.savefig('lr-data.png')

# Randomly initialize weights
m = np.random.randn()
b = np.random.randn()

learning_rate = 1e-6
for t in range(num_points):
    # Forward pass: compute predicted y
    y_pred = m * x + b

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(f'Iteration {t}, Loss: {loss:.2f}, m: {m:.3f}, b: {b:.3f}')

    # Backprop to compute gradients of m and b with respect to the loss.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_b = grad_y_pred.sum()
    grad_m = (grad_y_pred * x).sum()
    
    # Update weights
    m -= learning_rate * grad_m
    b -= learning_rate * grad_b
   
print(f'Model: y = {m:.3f}x + {b:.3f}')
x_data = 2.0
print(f'Prediction: y({x_data}) = {m * x_data + b:.3f}')