import torch
from torch.utils.data import DataLoader

from src.data import CustomDataset
from src.model import Static_reconstruction, CustomLoss

test_input_file = 'data/test_sdf_dataset'
test_target_file = 'data/test_observed_points_dataset'
model_path = 'out/model/model_50.pth'

test_dataset = CustomDataset(test_input_file, test_target_file)
# test_input_sample, test_target_sample = test_dataset[0]
# print("Test input sample:", test_input_sample)
# print("Test target sample:", test_target_sample)


batch_size = 100
input_size = 2500
hidden_size1 = 512
hidden_size2 = 128
output_size = 6
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Static_reconstruction(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(torch.load(model_path))
criterion = CustomLoss()  # Choose an appropriate loss function

test_loss = 0.0
num = 0
model.eval()
with open('data/test_sdf_pred', 'w') as file:
    pass
for i, data in enumerate(test_dataloader, 0):
    X_test, y_test = data
    # Rest of the code for training the neural network
    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)
    test_loss += loss.item()
    num += 1
    with open('data/test_sdf_pred', 'a') as file:
        for row in y_pred:
            row_str = ' '.join([str(element.item()) for element in row])
            file.write(row_str + '\n')

test_loss = test_loss / num
print("\nEvaluate the trained src on the X_test set: ")
print("loss: {:.5f}".format(test_loss))
