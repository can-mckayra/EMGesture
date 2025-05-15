import torch.nn as nn

class EMGCNN(nn.Module):
    def __init__(self, num_channels=10, num_classes=53):
        super(EMGCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),  # Conv1
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),           # Conv2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5, 128),   # 5 time steps left as input length 20 gets halved twice during the two pooling operations
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        # x shape: (batch, 10, 20)
        x = self.features(x)
        x = self.classifier(x)
        return x

model = EMGCNN()

from torchinfo import summary

summary(model,
        input_size=(1, 10, 20),    # (batch_size, num_channels, window_len)
        col_names=["input_size", "output_size", "num_params"],
        col_width=18, row_settings=["var_names"])
