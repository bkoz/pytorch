#
# Create a simple PyTorch linear regression model
# and save it to a file for use with nvidia triton.
#
import torch
from torch.autograd import Variable
import logging
import os

# Set up logging
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def concatenate_files(file_list, output_file):
    """
    Concatenate multiple files into a single output file.
    :param file_list: List of input file paths to concatenate.
    :param output_file: Output file path.
    """
    with open(output_file, 'w') as outfile:
        for filename in file_list:
            with open(filename, 'r') as infile:
                for line in infile:
                    outfile.write(line)

def create_model_repo():
    """
    Create a model repository for Triton Inference Server.
    This function creates a directory structure for the model
    and saves the model in the appropriate format.
    """
    model_repository = "models/lr"
    dir_path = model_repository + "/1"

    if os.path.isdir(dir_path):
        logger.info(f"Directory '{dir_path}' exists.")
    else:
        logger.info(f"Directory '{dir_path}' does not exist.")
        os.makedirs(dir_path)
        logger.info(f"Directory '{dir_path}' created.")

    # Save the model
    model_file = "/model.pt"
    model_file_path =  dir_path + model_file
    scripted_model = torch.jit.script(our_model)
    torch.jit.save(scripted_model, model_file_path)
    logger.info(f"Saved PyTorch Model State to {model_file_path}")

    # Save the model protobuf config file
    config_file = ['src/config.pbtxt']
    config_file_path = model_repository + '/config.pbtxt'
    concatenate_files(config_file, config_file_path)
    logger.info(f"Saved Triton Model Config to {config_file_path}")


class LinearRegressionModel(torch.nn.Module):
    """
    A simple linear regression model.
    This model takes a single input and produces a single output w/o a bias term.
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        Forward pass through the model.
        :param x: Input tensor.
        :return: Output tensor.
        """
        y_pred = self.linear(x)
        return y_pred

# Create the model instance.
our_model = LinearRegressionModel()

# The model is a linear regression model, so we can use
# the Mean Squared Error loss function
# and stochastic gradient descent as the optimizer.
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

# Training data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# Tell PyTorch to track the gradients of the input data
# so we can compute the gradients of the loss with respect to the input data.
x_data.requires_grad = True

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

# Test the model
new_var = Variable(torch.Tensor([[2.5]]))
pred_y = our_model(new_var)

logger.info(f"{new_var = } {pred_y = }")

create_model_repo()

