import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):
       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layers with gradual dimension reduction
       self.fc1 = nn.Linear(16 * 22050, 512)
       self.dropout1 = nn.Dropout(0.5)
       self.fc2 = nn.Linear(512, 128)
       self.dropout2 = nn.Dropout(0.5)
       self.fc3 = nn.Linear(128, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = F.relu(self.fc1(x))    # First FC layer with ReLU
       x = self.dropout1(x)       # Dropout for regularization
       x = F.relu(self.fc2(x))    # Second FC layer with ReLU
       x = self.dropout2(x)       # Dropout for regularization
       x = self.fc3(x)            # Final output layer
       return x