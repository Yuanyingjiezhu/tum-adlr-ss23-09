import numpy as np
import os
import glob
import pickle
from BPS_multitriangle import BPS

folder_path = 'data/whole_astar_trajectory'
multi_folder_path = 'data/whole_multi_trajectory'
# trajectory_Astar_files = glob.glob(os.path.join(folder_path, '*.npy'))
with open("data/list_data.pkl", "rb") as file:
    trajectory_Astar_files = pickle.load(file)
# print(trajectory_Astar_files)
trajectory_Astar_files = trajectory_Astar_files[31:32]
num_trajectory = len(trajectory_Astar_files)


num_timestep = 120
num_objects = 16
num_points = 50
x_range = (-1, 1)
y_range = (-1, 1)

time_interval = 1
extension = 10
# end_point = np.array([1, 1])
# start_point = np.array([-1, -1])

ground_truth_list = []
input_dataset_list = []


# generate ground truth
for file_path in trajectory_Astar_files:
    file_name = os.path.basename(file_path)
    print(file_name)
    astar_path = os.path.join(folder_path, file_name)
    print(astar_path)
    trajectory_Astar = np.load(astar_path)
    start_point = trajectory_Astar[0]
    end_point = trajectory_Astar[-1]
    trajectory_Astar = trajectory_Astar[1:]
    ground_truth = np.empty((trajectory_Astar.shape[0] // 3 + 1 + extension, 3, 2))

    for i in range(ground_truth.shape[0]):
        for j in range(3):
            if 3*i + j < trajectory_Astar.shape[0]:
                ground_truth[i][j] = trajectory_Astar[3*i + j]
            else:
                ground_truth[i][j] = end_point

    # file_name = os.path.basename(file_path)
    multi_path = os.path.join(multi_folder_path, file_name)
    print(multi_path)
    multi_trajectory = np.load(multi_path)
    Generator = BPS(x_range, y_range, num_points, multi_trajectory, num_objects)
    sdf_dataset = Generator.cal_sdf()

    dataset = np.empty((num_timestep - 2, 3, 2500))
    input_dataset = np.empty((num_timestep - 2, 3, 2500))

    # dataset = np.empty((num_timestep - 2, 2500))
    # input_dataset = np.empty((num_timestep - 2, 2500))


    for i in range(num_timestep - 2):
        for j in range(3):
            dataset[i][j] = sdf_dataset[i + j]

    for i in range(dataset.shape[0]):
        input_dataset[i][0] = dataset[i][2]
        input_dataset[i][1] = (dataset[i][2] - dataset[i][1]) / time_interval
        input_dataset[i][2] = (dataset[i][2] + dataset[i][0] - 2 * dataset[i][1]) / (time_interval ** 2)

    # for i in range(dataset.shape[0]):
    #     input_dataset[i][0] = dataset[i][2]
    #     input_dataset[i][1] = dataset[i][1]
    #     input_dataset[i][2] = dataset[i][0]

    input_dataset = input_dataset.reshape(input_dataset.shape[0], -1)
    input_dataset = input_dataset[: ground_truth.shape[0]]

    start_point_set = np.empty((ground_truth.shape[0], 2))
    start_point_set[0] = start_point
    start_point_set[1:] = ground_truth[:-1, -1, :]

    end_point_set = np.tile(end_point, (ground_truth.shape[0], 1))
    input_dataset = np.concatenate((input_dataset, start_point_set, end_point_set), axis=1)

    ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
    ground_truth_list.append(ground_truth)
    input_dataset_list.append(input_dataset)

total_input_dataset = np.concatenate(input_dataset_list, axis=0)
total_ground_truth = np.concatenate(ground_truth_list, axis=0)

np.save('data/all_data/test_whole_input_dataset_411', total_input_dataset)
np.save('data/all_data/test_whole_ground_truth_411', total_ground_truth)


########read sdf
# dynamic_pointset = []
# sdf_dataset = []
# with open('data/first_trajectory', 'r') as file:
#     for row in range(total_timestep):
#         line = file.readline().strip()
#         numbers = [float(num) for num in line.split()]
#         sdf_dataset.append(numbers)
# sdf_dataset = np.array(sdf_dataset)
##  shape of sdf_dataset is 120*2500


#############generate input
#############when overfitting, num_trajectory = 1

# dataset = np.empty((num_trajectory*(num_timestep-3), 3, 2500))
# input_dataset = np.empty((num_trajectory*(num_timestep-3), 3, 2500))
#
# for i in range(num_trajectory):
#     for j in range(num_timestep-3):
#         for k in range(3):
#             dataset[i*(num_timestep-3)+j][k] = np.array(sdf_dataset)[i*num_timestep+j+k]
## shape of dataset is 117*3*2500
# print(np.shape(input_dataset))

# for i in range(dataset.shape[0]):
#     input_dataset[i][0] = dataset[i][2]
#     input_dataset[i][1] = (dataset[i][2]-dataset[i][1])/time_interval
#     input_dataset[i][2] = (dataset[i][2]+dataset[i][0]-2*dataset[i][1])/time_interval**2
## shape of input_dataset is 117*3*2500

###############generate start point
# start_point_set = np.empty((ground_truth.shape[0], 2))
# start_point_set[0, :] = start_point
# start_point_set[1:, :] = ground_truth[:-1, 2, :]
# start_point_set[1:, :] = start_point_set[:-1, :]

# end_point_set = np.tile(end_point, (ground_truth.shape[0], 1))
# with open('data/start_point', 'w') as file:
#     for i in range(start_point.shape[0]):
#         for j in range(start_point.shape[1]):
#             file.write(str(start_point[i, j]) + " ")
#         file.write("\n")

# with open('data/input_dataset', 'w') as file:
#     for i in range(ground_truth.shape[0]):
#         for j in range(input_dataset.shape[1]):
#             for k in range(input_dataset.shape[2]):
#                 file.write(str(input_dataset[i, j, k]) + " ")
#         for m in range(2):
#             file.write(str(end_point_set[i, m]) + " ")
#         for m in range(2):
#             file.write(str(start_point_set[i, m]) + " ")
#         file.write("\n")






##########dividing into train&test dataset
# train_input_dataset = input_dataset[:2106, :, :]
# test_input_dataset = input_dataset[2106:, :, :]
#
# with open('data/train_input_dataset', 'w') as file:
#     for i in range(train_input_dataset.shape[0]):
#         for j in range(train_input_dataset.shape[1]):
#             for k in range(train_input_dataset.shape[2]):
#                 file.write(str(train_input_dataset[i, j, k]) + " ")
#         file.write("\n")
#
# with open('data/test_input_dataset', 'w') as file:
#     for i in range(test_input_dataset.shape[0]):
#         for j in range(test_input_dataset.shape[1]):
#             for k in range(test_input_dataset.shape[2]):
#                 file.write(str(test_input_dataset[i, j, k]) + " ")
#         file.write("\n")
