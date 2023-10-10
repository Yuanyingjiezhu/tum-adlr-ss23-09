import numpy as np
import matplotlib.pyplot as plt
import os

# Define the parameters for the triangles
num_triangles = 3

# Define the line coordinates
line_start = np.array([-1, -1])
line_end = np.array([1, 1])

# Define the circular trajectory parameters for each triangle
translation_speeds = [0.1, 0.15, 0.2]  # Translation speeds for each triangle
rotation_speeds = [0.02, 0.04, 0.06]  # Rotation speeds for each triangle
trajectory_sizes = [0.4, 0.4, 0.4]  # Sizes of the trajectories for each triangle
triangle_sizes = [0.5, 0.5, 0.5]  # Sizes of the triangles for each triangle
num_time_steps = 120  # Number of time steps for the circular trajectory

# Initialize the list to store the trajectories of the triangles
triangle_trajectories = []

for i in range(num_triangles):
    # Calculate the initial position of the triangle
    initial_position = line_start + (line_end - line_start) * (i + 1) / (num_triangles + 1)

    # Generate the circular trajectory for the current triangle
    theta = np.linspace(0, 2 * np.pi, num_time_steps)
    x_displacement = np.cos(theta) * translation_speeds[i]
    y_displacement = np.sin(theta) * translation_speeds[i]
    rotation_angles = theta * rotation_speeds[i]

    # Initialize the list to store the vertices of the triangle at each time step
    triangle_trajectory = []

    for j in range(num_time_steps):
        # Calculate the current position of the triangle
        current_position = initial_position + np.array([x_displacement[j], y_displacement[j]])

        # Calculate the size of the trajectory for the current triangle
        trajectory_size = trajectory_sizes[i]

        # Calculate the vertices of the triangle
        triangle_size = trajectory_size * triangle_sizes[i]  # Adjust the size of the triangle
        vertices = np.array([
            [current_position[0] - triangle_size / 2, current_position[1] - triangle_size / 2],
            [current_position[0] + triangle_size / 2, current_position[1] - triangle_size / 2],
            [current_position[0], current_position[1] + triangle_size / 2]
        ])

        # Apply rotation transformation
        rotation_matrix = np.array([
            [np.cos(rotation_angles[j]), -np.sin(rotation_angles[j])],
            [np.sin(rotation_angles[j]), np.cos(rotation_angles[j])]
        ])
        transformed_vertices = np.dot(vertices - current_position, rotation_matrix.T) + current_position

        # Append the vertices to the triangle trajectory
        triangle_trajectory.append(transformed_vertices)

    # Append the triangle trajectory to the list of trajectories
    triangle_trajectories.append(triangle_trajectory)

# Convert the triangle trajectories to the desired format (120, 3, 2)
triangle_trajectories = np.array(triangle_trajectories)

# Plot the trajectories of the triangles
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Circular Trajectories of Triangles')

for triangle_trajectory in triangle_trajectories:
    for vertices in triangle_trajectory:
        polygon = plt.Polygon(vertices, edgecolor='blue', fill=None)
        ax.add_patch(polygon)

plt.show()

save_dir = 'out/trajectory/'
save_file = 'ts1'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_file)
np.save(save_path, triangle_trajectories)

print(triangle_trajectories.shape)
