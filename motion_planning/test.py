import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from source.dataset import CustomDataset
from source.model import MotionPlanning


test_input_file = 'data/all_data/test_whole_input_dataset_11.npy'
test_target_file = 'data/all_data/test_whole_ground_truth_11.npy'
model_path = 'out/model/whole_motion_planning_best_11_new.pth'

test_dataset = CustomDataset(test_input_file, test_target_file)
# test_input_sample, test_target_sample = test_dataset[0]
# print("Test input sample:", np.shape(test_input_sample))
# print("Test target sample:", np.shape(test_target_sample))

batch_size = 1
input_size = 7500
hidden_size1 = 2048
hidden_size2 = 128
hidden_size3 = 64
output_size = 6
# num_epochs = 1000
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = MotionPlanning(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
model.load_state_dict(torch.load(model_path))
# criterion = nn.MSELoss()  # Choose an appropriate loss function

# test_loss = 0.0
# num = 0
model.eval()
# trajectory = []
# with open('data/test_output', 'w') as file:
#     pass

for i, data in enumerate(test_dataloader, 0):
    # print(i)
    X_test, y_test = data
    if i != 0:
        X_test[:, 7500:7502] = end
    # Rest of the code for training the neural network
    y_pred = model(X_test)
    tr = y_pred.reshape(3, 2)
    if i == 0:
        trajectory = tr
        # print(trajectory)
    else:
        trajectory = torch.cat((trajectory, tr), axis=0)
    # print(trajectory.shape)
    end = y_pred[:, 5:7]

trajectory = trajectory.detach().numpy()
print(trajectory.shape)

np.save('data/output/test_whole_out_11_new_new', trajectory)
    # loss = criterion(y_pred, y_test)
    # test_loss += loss.item()
    # num += 1
    # with open('data/test_output', 'a') as file:
    #     for row in y_pred:
    #         row_str = ' '.join([str(element.item()) for element in row])
    #         file.write(row_str + '\n')

# test_loss = test_loss / num
# print("loss: {:.5f}".format(test_loss))
