import numpy as np

first_trajectory = []
with open('data/first_trajectory', 'r') as file:
    for row in range(120):
        line = file.readline().strip()
        numbers = [float(num) for num in line.split()]
        first_trajectory.append(numbers)
first_trajectory = np.array(first_trajectory)
print(np.shape(first_trajectory))
num_zeros = np.count_nonzero(first_trajectory == 0)
print("Number of zeros in the 2D array:", num_zeros)
num_negatives = np.count_nonzero(first_trajectory < 0)
print("Number of negative values in the 2D array:", num_negatives)
