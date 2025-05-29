import torch.nn as nn

class EMGesture(nn.Module):

    """Expects input of shape (N, 1, 15, 10) = (batch, channel, time, electrodes)"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_electrodes = 10
        self.time_steps = 15 # 150 ms @ 100 Hz
        self.relu = nn.ReLU()

        # Block 1 (N, 32, 15, 1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, self.num_electrodes))

        # Block 2 (N, 32, 15, 1) -> pool (N, 32, 7, 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))

        # Block 3 (N, 64, 7, 1) -> pool (N, 64, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0))
        self.pool2 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))

        # Block 4 (N, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0))

        # Block 5 classifier
        self.adapt = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x)); x = self.pool1(x)
        x = self.relu(self.conv3(x)); x = self.pool2(x)
        x = self.relu(self.conv4(x))
        x = self.conv5(self.adapt(x))
        return x.view(x.size(0), -1) # (N, num_classes)

"""
Text description of the model (taken from the paper):

The first block of the net is composed of the following
parts. First, it includes a convolutional layer composed of 32
filters. After several tests, including different shapes and sizes,
the filters were defined as a row of the length of number of
electrodes. Second, it includes a rectified linear unit as a non-
linear activation function.

The second block of the net is composed of the following
three parts. The first one is a convolutional layer with 32 filters
of size 3 × 3. The second one is a non-linear activation function
(rectified linear unit). The third one is a subsampling layer that
performs an average pooling with filters of size 3 × 3.

The third block of the net is composed of the following three
parts. The first one is a convolutional layer with 64 filters of
size 5 × 5. The second one is a non linear activation function
(rectified linear unit). The third one is a subsampling layer that
performs an average pooling with filters of size 3 × 3.

The fourth block of the net is composed of the following two
parts. The first is a convolutional layer with 64 filters of size
5 × 1 for the Otto Bock electrodes and size 9 × 1 for the Delsys
electrodes. The second is a rectified linear unit.

The fifth block of the net is composed of the following two
parts. The first one is a convolutional layer with filters of size
1 × 1. The second is a softmaxloss.

Block 1:
Input: (N, 1, 15, 10)
Output: (N, 32, 15, 1)
32 filters of size 1 × 10
ReLU

Block 2:
Input: (N, 32, 15, 1)
Output: (N, 32, 7, 1)
32 filters of size 3 × 1
ReLU
Average Pooling (kernel size 3, stride 2)

Block 3:
Input: (N, 32, 7, 1)
Output: (N, 64, 3, 1)
64 filters of size 5 × 1
ReLU
Average Pooling (kernel size 3, stride 2)

Block 4:
Input: (N, 64, 3, 1)
Output: (N, 64, 1, 1)
64 filters of size 5 × 1
ReLU

Block 5:
Input: (N, 64, 1, 1)
Output: (N, num_classes, 1, 1)
1 filter of size 1 × 1
"""
