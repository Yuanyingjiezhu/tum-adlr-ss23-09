import torch.nn as nn
import torch


class MotionPlanning(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MotionPlanning, self).__init__()
        # super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.activation2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2+4, hidden_size3)
        self.activation3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        out = self.activation1(self.bn1(self.fc1(x[:, : 7500])))
        out = self.activation2(self.bn2(self.fc2(out)))
        # Concatenate the additional input with the output ocf fc2
        out = torch.cat([out, x[:, 7500:7504]], dim=1)
        out = self.activation3(self.fc3(out))
        out = self.fc4(out)
        return out
