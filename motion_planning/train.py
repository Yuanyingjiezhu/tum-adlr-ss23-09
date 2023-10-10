import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from source.dataset import CustomDataset
from source.model import MotionPlanning

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
input_file = 'data/all_data/test_whole_input_dataset_11.npy'
# val_input_file = 'data/all_data/val_whole_input_dataset.npy'
target_file = 'data/all_data/test_whole_ground_truth_11.npy'
# val_target_file = 'data/all_data/val_whole_ground_truth.npy'
save_dir = 'out/model/'
save_file = 'whole_motion_planning_11_new.pth'
save_file_best = 'whole_motion_planning_best_11_new.pth'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_file)
save_path_best = os.path.join(save_dir, save_file_best)
batch_size = 16

dataset = CustomDataset(input_file, target_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# val_dataset = CustomDataset(val_input_file, val_target_file)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = 7500
hidden_size1 = 2048
hidden_size2 = 128
hidden_size3 = 64
output_size = 6
num_epochs = 100
best_loss = float('inf')
# Define your neural network src
model = MotionPlanning(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
criterion = nn.MSELoss()  # Choose an appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.000001)  # Choose an optimizer
train_loss_history = []  # loss
model.train()

for epoch in range(num_epochs):

    running_loss = 0.0

    # Iterating through the minibatches of the data

    for i, data in enumerate(dataloader, 0):

        # data is a tuple of (inputs, labels)
        X, y = data
        # print(X.shape)
        # print(y.shape)

        # Reset the parameter gradients  for the current  minibatch iteration
        optimizer.zero_grad()

        y_pred = model(X)  # Perform a forward pass on the network with inputs
        loss = criterion(y_pred, y)  # calculate the loss with the network predictions and ground Truth
        loss.backward()  # Perform a backward pass to calculate the gradients
        optimizer.step()  # Optimize the network parameters with calculated gradients

        # Accumulate the loss and calculate the accuracy of predictions
        running_loss += loss.item()

        # Print statistics to console
        if i % 5 == 4:  # print every 100 mini-batches
            running_loss /= 5
            print("[Epoch %d, Iteration %5d] loss: %.5f " % (epoch + 1, i + 1, running_loss))
            train_loss_history.append(running_loss)
            if running_loss < best_loss:
                print(f"Running loss improved: {best_loss:.5f} -> {running_loss:.5f}. Saving model...")
                best_loss = running_loss
                torch.save(model.state_dict(), save_path_best)
            running_loss = 0.0

    # model.eval()  # Set the model to evaluation mode
    # val_loss = 0.0

    # with torch.no_grad():
    #     for val_data in val_dataloader:
    #         val_X, val_y = val_data
    #         val_pred = model(val_X)
    #         val_loss += criterion(val_pred, val_y).item()
    #
    # val_loss /= len(val_dataloader)
    #
    # # Print statistics and update best model
    # if val_loss < best_val_loss:
    #     print(f"Validation loss improved: {best_val_loss:.5f} -> {val_loss:.5f}. Saving model...")
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), save_path_best)
    # print(f"[Epoch {epoch + 1}] Train loss: {running_loss:.5f} | Validation loss: {val_loss:.5f}")

torch.save(model.state_dict(), save_path)
print('FINISH.')
# plt.plot(train_loss_history)
# plt.title("sdf")
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend(['loss'])
# plt.show()

plt.plot(train_loss_history)
plt.title("sdf")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['loss'])
plt.savefig('out/whole_loss_plot_11_new.png')
plt.close()
