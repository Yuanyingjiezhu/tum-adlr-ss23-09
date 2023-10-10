import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

for i in range(40):
    save_dir = 'out/trajectory/'
    save_file = '3_1_' + str(i+1)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_file)

    x = np.random.uniform(-0.1, 0.1, size=3)
    y = np.random.uniform(-0.1, 0.1, size=3)
    vertices = np.column_stack((x, y))

    triangle = plt.Polygon(vertices, closed=True, fill=None)
    fig, ax = plt.subplots()
    ax.add_patch(triangle)
    ax.set_aspect('equal')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Triangle')
    plt.show()

    # Define the initial mesh (triangle)
    # vertices = np.array([
    #     [-0.1, -0.1],  # Vertex 0
    #     [0.1, -0.1],  # Vertex 1
    #     [0, 0],  # Vertex 2
    # ])

    max_x = np.max(vertices[:, 0])
    min_x = np.min(vertices[:, 0])
    max_y = np.max(vertices[:, 1])
    min_y = np.min(vertices[:, 1])

    # Define the square trajectory
    num_time_steps = 30  # Number of time steps on each edge
    side_length = 0.15  # Side length of the square

    # Calculate the x and y displacements for the square trajectory
    x_displacement = np.concatenate([
        np.linspace(-side_length / 2, side_length / 2, num_time_steps),  # Bottom edge
        np.ones(num_time_steps) * side_length / 2,  # Right edge
        np.linspace(side_length / 2, -side_length / 2, num_time_steps),  # Top edge
        np.ones(num_time_steps) * -side_length / 2  # Left edge
    ])
    x_displacement = np.clip(x_displacement, -1 - min_x, 1 - max_x)

    # print(x_displacement)
    y_displacement = np.concatenate([
        np.ones(num_time_steps) * -side_length / 2,  # Bottom edge
        np.linspace(-side_length / 2, side_length / 2, num_time_steps),  # Right edge
        np.ones(num_time_steps) * side_length / 2,  # Top edge
        np.linspace(side_length / 2, -side_length / 2, num_time_steps)  # Left edge
    ])
    y_displacement = np.clip(y_displacement, -1 - min_y, 1 - max_y)
    # print(y_displacement)

    # Define the rotation speed
    rotation_speed = 0.15  # Fixed rotation speed

    # # Define the velocity modulation parameters
    # min_velocity = 0.8   # Minimum velocity
    # max_velocity = 1.2   # Maximum velocity
    # velocity_factor = np.linspace(min_velocity, max_velocity, num_time_steps * 4)  # Time-varying scaling factor

    # Define the time-varying translation speed
    min_translation_speed = 0.4  # Minimum translation speed
    max_translation_speed = 0.8  # Maximum translation speed
    translation_speed_factor = np.linspace(min_translation_speed, max_translation_speed, num_time_steps * 4)


    # Initialize a list to store the history of mesh transformations
    mesh_history = []

    # Apply transformations to the vertices at each time step
    for i in range(num_time_steps * 4):
        translation = np.array([[x_displacement[i]], [y_displacement[i]], [0]])  # Translation matrix

        # Create the transformation matrix for translation
        transformation_matrix = np.eye(3)
        # transformation_matrix[:2, 2] = translation[:2, 0]
        transformation_matrix[:2, 2] = translation[:2, 0] * translation_speed_factor[i]  # Apply translation speed

        # Apply rotation transformation
        rotation_angle = i * rotation_speed  # Calculate the rotation angle based on the time step
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        transformation_matrix = np.dot(rotation_matrix, transformation_matrix)

        # # Apply velocity modulation to the transformation matrix
        # velocity_scaling = np.diag([velocity_factor[i], velocity_factor[i], 1])
        # transformation_matrix = np.dot(velocity_scaling, transformation_matrix)

        # Apply the transformation matrix to the vertices
        augmented_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        augmented_transformed_vertices = np.dot(augmented_vertices, transformation_matrix.T)
        transformed_vertices = augmented_transformed_vertices[:, 0:2]
        # print(transformed_vertices)
        # transformed_vertices[:, 0] += 0.7
        # transformed_vertices[:, 0] += 0.2
        # transformed_vertices[:, 0] -= 0.3
        transformed_vertices[:, 0] -= 0.8
        # print(transformed_vertices)

        # transformed_vertices[:, 1] += 0.7
        # transformed_vertices[:, 1] += 0.2
        transformed_vertices[:, 1] -= 0.3
        # transformed_vertices[:, 1] -= 0.8

        # Append the transformed vertices to the mesh history
        mesh_history.append(transformed_vertices)

    # Plot the history of the mesh
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Square Trajectory with Rotation of Triangle Mesh')

    for transformed_vertices in mesh_history:
        polygon = plt.Polygon(transformed_vertices[:, :2], edgecolor='blue', fill=None)
        ax.add_patch(polygon)

    plt.show()
    # plt.close()

    # Query for a specific time step
    # query_time_step = 1  # Time step to query
    #
    # # Retrieve the transformed vertices for the query time step
    # query_vertices = mesh_history[query_time_step]
    #
    # # Print the transformed vertices for the query time step
    # print(f"Vertices at time step {query_time_step}:\n{query_vertices}")
    # print(np.shape(mesh_history))
    # print(np.max(mesh_history))
    # print(np.min(mesh_history))
    np.save(save_path, mesh_history)




# fig, ax = plt.subplots()
#
# # Set axis limits
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
#
# # Initialize an empty plot
# line, = ax.plot([], [], 'b-')
#
# # Update function for the animation
# def update(frame):
#     # Clear the current plot
#     ax.cla()
#
#     # Set axis limits
#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#
#     # Get the vertices for the current frame
#     vertices = mesh_history[frame]
#
#     # Update the plot with the new vertices
#     line.set_data(vertices[:, 0], vertices[:, 1])
#
#     return line,
#
# # Create the animation
# animation = FuncAnimation(fig, update, frames=len(mesh_history), interval=200)
# plt.show()
# # Save the animation as a GIF
# # animation.save('mesh_animation.gif', writer='pillow')
# #
# # # Close the plot window
# # plt.close()




# # Create a list to store the images for the GIF
# images = []
#
# # Plot the history of the mesh and save each frame as an image
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect('equal')
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Square Trajectory with Rotation of Triangle Mesh')
#
# for transformed_vertices in mesh_history:
#     # ax.clear()
#     # ax.set_aspect('equal')
#     # ax.set_xlim(-1.5, 1.5)
#     # ax.set_ylim(-1.5, 1.5)
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_title('Square Trajectory with Rotation of Triangle Mesh')
#     polygon = plt.Polygon(transformed_vertices[:, :2], edgecolor='blue', fill=None)
#     ax.add_patch(polygon)
#
#     # Save the current frame as an image
#     fig.canvas.draw()
#     image = np.array(fig.canvas.renderer.buffer_rgba())
#     images.append(image)
#
#     plt.pause(1)
#
# # Save the images as a GIF
# imageio.mimsave('mesh_animation.gif', images, duration=0.1)
#
# # Display the GIF
# from IPython.display import Image
#
# Image(filename='mesh_animation.gif')