import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.dataset import CustomDataset
from src.model import Dynamic_reconstruction
import numpy as np

test_input_file = 'data/test_input_dataset'
test_target_file = 'data/test_ground_truth'
model_path = '../out/model/dynamic_model.pth'

test_dataset = CustomDataset(test_input_file, test_target_file)
test_input_sample, test_target_sample = test_dataset[0]
# print("Test input sample:", np.shape(test_input_sample))
# print("Test target sample:", np.shape(test_target_sample))

batch_size = 16
input_size = 7500
hidden_size1 = 6000
hidden_size2 = 4096
output_size = 2500
num_epochs = 50
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = Dynamic_reconstruction(input_size, hidden_size1, hidden_size2,  output_size)
model.load_state_dict(torch.load(model_path))
criterion = nn.MSELoss()  # Choose an appropriate loss function

test_loss = 0.0
num = 0
model.eval()
with open('data/test_output', 'w') as file:
    pass
for i, data in enumerate(test_dataloader, 0):
    X_test, y_test = data
    # Rest of the code for training the neural network
    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)
    test_loss += loss.item()
    num += 1
    with open('data/test_output', 'a') as file:
        for row in y_pred:
            row_str = ' '.join([str(element.item()) for element in row])
            file.write(row_str + '\n')

test_loss = test_loss / num
print("loss: {:.5f}".format(test_loss))
