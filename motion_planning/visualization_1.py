import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.animation as animation


def is_point_inside_triangle(point, triangle_vertices):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    for triangle in triangle_vertices:
        v1, v2, v3 = triangle
        d1 = sign(point, v1, v2)
        d2 = sign(point, v2, v3)
        d3 = sign(point, v3, v1)

        has_neg = np.any([d1 < 0, d2 < 0, d3 < 0])
        has_pos = np.any([d1 > 0, d2 > 0, d3 > 0])

        if not (has_neg and has_pos):
            return True

    return False


def animate(i):
    # agent_position = trajectory[i]
    # scat_agent.set_offsets(agent_position)
    triangle_step_index = i // 3
    for j, line_triangle in enumerate(lines_triangles):
        triangle_vertices = triangle_meshes[j, triangle_step_index, :, :]
        triangle_vertices = np.vstack((triangle_vertices, triangle_vertices[0]))  # Close the triangle
        line_triangle.set_data(triangle_vertices[:, 0], triangle_vertices[:, 1])
    return scat_agent, *lines_triangles


# def animate(i):
#     agent_position = trajectory[i]
#     scat_agent.set_offsets(agent_position)
#     triangle_step_index = i
#     for j, line_triangle in enumerate(lines_triangles):
#         triangle_vertices_1 = triangle_meshes[j, triangle_step_index, :, :]
#         triangle_vertices_2 = triangle_meshes[j, triangle_step_index + 1, :, :]
#         triangle_vertices_1 = np.vstack((triangle_vertices_1, triangle_vertices_1[0]))  # Close the first triangle
#         triangle_vertices_2 = np.vstack((triangle_vertices_2, triangle_vertices_2[0]))  # Close the second triangle
#         triangle_vertices = np.vstack((triangle_vertices_1, triangle_vertices_2))  # Stack both triangles
#         line_triangle.set_data(triangle_vertices[:, 0], triangle_vertices[:, 1])
#     return scat_agent, *lines_triangles



# trajectory = np.load('data/output/test_whole_out_11.npy')
trajectory = np.load('data/whole_astar_trajectory/T00.npy')
trajectory = trajectory[1:, :]
obstacles = np.load('../out/whole_multi_trajectory/T00.npy')
# desired_indices = np.arange(2, whole_trajectory.shape[0], 3)
# trajectory = whole_trajectory[desired_indices]
triangle_meshes = obstacles[:, 2: 4 + trajectory.shape[0] // 3, :, :]  # Replace this with your actual triangle meshes
# triangle_meshes = obstacles
fig, ax = plt.subplots()
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
scat_agent = ax.scatter([], [], color='blue')
lines_triangles = [ax.plot([], [], color='red', linestyle='-', linewidth=2)[0]
                   for _ in range(triangle_meshes.shape[0])]

ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=len(trajectory), interval=50)
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
# plt.title('Animation of Agent Trajectory and Obstacles')
plt.title('Dynamic Environment')
# Save the animation using Pillow as a GIF
writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('data/output/dynamic_00_new.gif', writer=writer)
plt.show()

x_values = trajectory[:, 0]
y_values = trajectory[:, 1]
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory')
# plt.savefig('data/output/astar_33_double.png')
# Show the plot
plt.show()


# Separate the x and y coordinates of the points
for i in range(trajectory.shape[0]//3):
    triangles_vertices = obstacles[:, 2+i, :, :]
    points = trajectory[i*3:(i+1)*3, :]
    x_values = points[0:1, 0]
    y_values = points[0:1, 1]
    for point in points:
        result = is_point_inside_triangle(point, triangles_vertices)
        if result is True:
            print(f"Point {point}: Inside any triangle? {result}")
    # if i % 4 == 0:
    #     fig, ax = plt.subplots()
    #     triangles = PolyCollection(triangles_vertices, edgecolors='blue', facecolors='none')
    #     ax.add_collection(triangles)
    #     # Plot the trajectory
    #     plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    #     # Add labels and title
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Trajectory')
    #     # Show the plot
    #     plt.show()
