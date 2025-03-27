from torch import nn

# Define model
class NeuralNetwork(nn.Module):
    """
    Define a network to classify mnist images.

    Args:
        nn (nn.Module): Parent Class
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Run forward and make an inference.

        Args:
            x (tensor): input tensor

        Returns:
            logits: tensor
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits