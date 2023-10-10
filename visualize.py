import numpy as np
import matplotlib.pyplot as plt


num_test_data = 1000

# Read the file contents
with open('data/test_sdf_pred', 'r') as file:
    lines = file.readlines()

# Create an empty array to store the data
test_sdf_pred = np.zeros((num_test_data, 6))

# Iterate over each line and extract the values
for i, line in enumerate(lines):
    values = line.strip().split()
    test_sdf_pred[i] = np.array(values, dtype=np.float32)


with open('data/test_observed_points_dataset', 'r') as file:
    lines = file.readlines()

# Create an empty array to store the data
observed_points_dataset = np.zeros((num_test_data, 6))

# Iterate over each line and extract the values
for i, line in enumerate(lines):
    values = line.strip().split()
    observed_points_dataset[i] = np.array(values, dtype=np.float32)


for i in range(len(test_sdf_pred)):
    # Get the current row from test_sdf_pred and observed_points_dataset
    test_row = test_sdf_pred[i]
    observed_row = observed_points_dataset[i]

    # Reshape the rows into 3x2 matrices representing the 3 points
    test_points = test_row.reshape(3, 2)
    observed_points = observed_row.reshape(3, 2)

    # Extract x and y coordinates for each point in test_sdf_pred
    test_x_coords = test_points[:, 0]
    test_y_coords = test_points[:, 1]

    # Extract x and y coordinates for each point in observed_points_dataset
    observed_x_coords = observed_points[:, 0]
    observed_y_coords = observed_points[:, 1]

    # Create a new plot for the current row
    plt.figure()

    # Plot the points and lines for test_sdf_pred
    plt.plot(test_x_coords, test_y_coords, 'ro')
    plt.plot(test_x_coords, test_y_coords, 'b-')

    # Plot the points and lines for observed_points_dataset
    plt.plot(observed_x_coords, observed_y_coords, 'go')
    plt.plot(observed_x_coords, observed_y_coords, 'm-')

    # Set plot title and labels
    plt.title(f"Points and Connecting Lines - Row {i}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Show the plot
    plt.show()
