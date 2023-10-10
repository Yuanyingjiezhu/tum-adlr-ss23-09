import os

import numpy as np

T = []
traj_dir = 'out/trajectory/'
multi_traj_dir = 'out/multi_trajectory/'

start = 173

file_extension = '.npy'

save_dir = 'out/whole_multi_trajectory/'

os.makedirs(save_dir, exist_ok=True)


for i in range(10):
    original_file = 'T3' + str(i) + str(i) + file_extension
    original_path = os.path.join(multi_traj_dir, original_file)
    T_1 = np.load(original_path)
    for m in range(4):
        T.append(T_1[m])
    for j in range(4):
        for k in range(4):
            if j != k:
                if j + k + 2 == 5:
                    file_name = str(j + 1) + '_' + str(k + 1) + '_' + str(i + 11) + file_extension
                else:
                    file_name = str(j+1) + '_' + str(k+1) + '_' + str(i+31) + file_extension
                file_path = os.path.join(traj_dir, file_name)
                T_0 = np.load(file_path)
                T.append(T_0)
    save_file = original_file
    save_path = os.path.join(save_dir, save_file)
    T = np.array(T)
    np.save(save_path, T)
    T = []

# for i in range(4):
#
#     file_name =  + str(file_index) + file_extension
#     file_path = os.path.join(traj_dir, file_name)
#     T_0 = np.load(file_path)
#     T.append(T_0)
# T = np.array(T)
# np.save(save_path, T)
