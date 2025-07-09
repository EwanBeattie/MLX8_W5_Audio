import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, in_channels=1, num_classes=10, dropout=0.1):
       """
       2D CNN for mel spectrogram classification.

       Parameters:
           * in_channels: Number of input channels (1 for single-channel spectrograms)
           * num_classes: Number of classes to predict (10 for UrbanSound8K)
           * dropout: Dropout rate for regularization
       """
       super(CNN, self).__init__()

       # 1st convolutional block
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
       self.bn1 = nn.BatchNorm2d(32)
       self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
       # 2nd convolutional block
       self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
       self.bn2 = nn.BatchNorm2d(64)
       self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
       # 3rd convolutional block - no pooling to preserve more spatial info
       self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
       self.bn3 = nn.BatchNorm2d(128)
       
       # Global Average Pooling instead of aggressive pooling
       self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
       
       # Fully connected layers
       self.fc1 = nn.Linear(128, 64)
       self.dropout1 = nn.Dropout(dropout)
       self.fc2 = nn.Linear(64, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor of shape (batch_size, channels, n_mels, time)

       Returns:
           torch.Tensor: Output tensor of shape (batch_size, num_classes)
       """
       # 1st convolutional block
       x = F.relu(self.bn1(self.conv1(x)))
       x = self.pool1(x)
       
       # 2nd convolutional block
       x = F.relu(self.bn2(self.conv2(x)))
       x = self.pool2(x)
       
       # 3rd convolutional block (no pooling)
       x = F.relu(self.bn3(self.conv3(x)))
       
       # Global average pooling to preserve spatial features
       x = self.global_pool(x)  # Output: [batch, 128, 1, 1]
       x = x.view(x.size(0), -1)  # Flatten: [batch, 128]
       
       # Fully connected layers
       x = F.relu(self.fc1(x))
       x = self.dropout1(x)
       x = self.fc2(x)
       
       return x