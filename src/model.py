import torch
import torch.nn as nn
from torch import Tensor


class Static_reconstruction(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Static_reconstruction, self).__init__()
        # super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.atvn1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.atvn2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # self.bn3 = nn.BatchNorm1d(hidden_size3)
        # self.atvn3 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.atvn1(self.bn1(self.fc1(x)))
        out = self.atvn2(self.bn2(self.fc2(out)))
        # out = self.atvn3(self.bn3(self.fc3(out)))
        out = self.fc3(out)
        return out


class CustomLoss(nn.Module):
    # compute the Chamfer Distance, i.e. the dissimilarity between two point sets
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Reshape the tensors to have shape (2, 3), representing the coordinates of the vertices
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_pred = y_pred.reshape(-1, 2)
        y_true = y_true.reshape(-1, 2)
        # Calculate the pairwise Euclidean distances between the points in y_pred and y_true
        distances = torch.cdist(y_pred, y_true, p=2)
        # Compute the minimum distance from each point in y_pred to any point in y_true
        min_distances_pred_to_true = torch.min(distances, dim=1)[0]
        # Compute the minimum distance from each point in y_true to any point in y_pred
        min_distances_true_to_pred = torch.min(distances, dim=0)[0]
        # Compute the total loss as the sum of the minimum distances in both directions
        total_loss = torch.sum(min_distances_pred_to_true) + torch.sum(min_distances_true_to_pred)
        loss = total_loss / y_true.shape[0]
        return loss

