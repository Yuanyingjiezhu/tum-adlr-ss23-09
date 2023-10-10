import numpy as np
import os

# Specify the directory containing the .npy files
directory = 'out/trajectory/'

# Get a list of all .npy files in the directory
file_list = sorted([file for file in os.listdir(directory) if file.endswith('.npy')])

# Load the .npy files and concatenate them
concatenated_array = np.concatenate([np.load(os.path.join(directory, file)) for file in file_list], axis=0)

# Save the concatenated array as a new .npy file
np.save('concatenated.npy', concatenated_array)

# Verify the shape of the concatenated array
print(concatenated_array.shape)  # (2400, 3, 2)