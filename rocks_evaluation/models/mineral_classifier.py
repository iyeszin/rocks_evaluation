import torch
import torch.nn as nn

n_class = 14

class SimpleCNN1D(nn.Module):
    def __init__(self, n_class=n_class):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (1000 // 4), 100)
        self.fc2 = nn.Linear(100, n_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class UncertaintyAwareCNN1D(nn.Module):
    def __init__(self, n_class=n_class, dropout_rate=0.3):
        super(UncertaintyAwareCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32 * (1000 // 4), 100)
        self.fc2 = nn.Linear(100, n_class)

    def forward(self, x, enable_dropout=False):
        if enable_dropout:
            self.train()
        else:
            self.eval()
            
        x = x.unsqueeze(1)
        x = self.dropout1(torch.relu(self.conv1(x)))
        x = torch.max_pool1d(x, kernel_size=2)
        x = self.dropout2(torch.relu(self.conv2(x)))
        x = torch.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = self.dropout3(torch.relu(self.fc1(x)))
        return self.fc2(x)