# Create a simple pytorch linear regression model
# and save it to a file for use with nvidia triton.
#
import torch
from torch.autograd import Variable
import logging
import os

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Training data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# our model
our_model = LinearRegressionModel()

# The model is a linear regression model, so we can use
# the Mean Squared Error loss function
# and stochastic gradient descent as the optimizer.
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

# Training loop
epochs = 100
for epoch in range(epochs):

    # Forward pass: Compute predicted y by passing 
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass, 
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.debug('epoch {}, loss {}'.format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[2.5]]))
pred_y = our_model(new_var)

# logger.info(f"{x_data = } {y_data}")
logger.info(f"{new_var = } {pred_y = }")

# Save the model
dir_path = "models"

if os.path.isdir(dir_path):
    logger.info(f"Directory '{dir_path}' exists.")
else:
    logger.info(f"Directory '{dir_path}' does not exist.")
    os.mkdir("models")

model_file = "/lr.pt"
dir_path = "models" + model_file
# torch.save(our_model.state_dict(), dir_path)
logger.info(f"Saved PyTorch Model State to {dir_path}")
scripted_model = torch.jit.script(our_model)
torch.jit.save(scripted_model, dir_path)