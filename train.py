import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchsummaryX import summary
from torchviz import make_dot

from src.data import CustomDataset
from src.model import Static_reconstruction, CustomLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_input_file = 'data/sdf_dataset'
train_target_file = 'data/observed_points_dataset'
eval_input_file = 'data/eval_sdf_dataset'
eval_target_file = 'data/eval_observed_points_dataset'
save_dir = 'out/model/'
save_file = 'model_50_3.pth'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_file)


train_dataset = CustomDataset(train_input_file, train_target_file)
eval_dataset = CustomDataset(eval_input_file, eval_target_file)
# input_sample, target_sample = dataset[0]
# print("Input sample:", input_sample)
# print("Target sample:", target_sample)


batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size= batch_size, shuffle=False)

input_size = 2500
hidden_size1 = 512
hidden_size2 = 128
output_size = 6
num_epochs = 5000
# Define your neural network src
model = Static_reconstruction(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(torch.load('out/model/model_50_2.pth'))
criterion = CustomLoss()  # Choose an appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)  # Choose an optimizer
train_loss_history = []  # loss
eval_loss_history = []

# dummy_input = torch.randn(batch_size, input_size)
# summary(model, dummy_input)
# output = model(dummy_input)
# graph = make_dot(output, params=dict(model.named_parameters()))
# graph.render("model_graph")


for epoch in range(num_epochs):

    running_loss = 0.0

    # Iterating through the minibatches of the data

    for i, data in enumerate(train_dataloader, 0):

        # data is a tuple of (inputs, labels)
        X, y = data

        # Reset the parameter gradients  for the current  minibatch iteration
        model.train()
        optimizer.zero_grad()

        y_pred = model(X)  # Perform a forward pass on the network with inputs
        loss = criterion(y_pred, y)  # calculate the loss with the network predictions and ground Truth
        loss.backward()  # Perform a backward pass to calculate the gradients
        optimizer.step()  # Optimize the network parameters with calculated gradients

        # Accumulate the loss and calculate the accuracy of predictions
        running_loss += loss.item()

        # Print statistics to console
        if i % 10 == 9:  # print every 5 mini-batches
            running_loss /= 10
            print("[Epoch %d, Iteration %5d] loss: %.3f " % (epoch + 1, i + 1, running_loss))
            train_loss_history.append(running_loss)
            running_loss = 0.0

            model.eval()  # Set the model to evaluation mode
            eval_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_data in eval_dataloader:
                    val_X, val_y = val_data
                    val_y_pred = model(val_X)
                    val_loss = criterion(val_y_pred, val_y)
                    eval_loss += val_loss.item()

            eval_loss /= len(eval_dataloader)
            print("[Epoch %d, Iteration %5d] Validation loss: %.3f" % (epoch + 1, i + 1, eval_loss))
            eval_loss_history.append(eval_loss)
torch.save(model.state_dict(), save_path)
print('FINISH.')
# plt.plot(train_loss_history)
# plt.title("sdf")
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend(['loss'])
# plt.show()

# plt.plot(train_loss_history)
# plt.title("static reconstruction loss")
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend(['loss'])
# plt.savefig('out/loss_plot_50.png')
# plt.close()

# Plot the loss curves
plt.figure()
plt.plot(train_loss_history, label='Training Loss')
plt.plot(eval_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the figure
plt.savefig('out/loss_plot_50_3.png')
