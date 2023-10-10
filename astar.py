import os

import numpy as np
from matplotlib import pyplot as plt


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            if Node(current_node, node_position) in closed_list:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                # print(child)
                # print(closed_child)
                if child == closed_child:
                    break

            else:
                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g >= open_node.g:
                        break

                else:
                    # Add the child to the open list
                    open_list.append(child)


# def is_point_in_triangle(x, y, triangle):
#     x1, y1 = triangle[0]
#     x2, y2 = triangle[1]
#     x3, y3 = triangle[2]
#     area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
#     s = 1 / (2 * area) * (y1 * x3 - x1 * y3 + (y3 - y1) * x + (x1 - x3) * y)
#     t = 1 / (2 * area) * (x1 * y2 - y1 * x2 + (y1 - y2) * x + (x2 - x1) * y)
#     return s >= 0 and t >= 0 and (1 - s - t) >= 0


def is_point_in_triangle(x, y, triangle, scale):
    centroid_x = (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3
    centroid_y = (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3
    translated_triangle = [(x - centroid_x, y - centroid_y) for x, y in triangle]
    scaled_triangle = [(scale * x, scale * y) for x, y in translated_triangle]
    new_triangle = [(x + centroid_x, y + centroid_y) for x, y in scaled_triangle]
    x1, y1 = new_triangle[0]
    x2, y2 = new_triangle[1]
    x3, y3 = new_triangle[2]
    area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
    s = 1 / (2 * area) * (y1 * x3 - x1 * y3 + (y3 - y1) * x + (x1 - x3) * y)
    t = 1 / (2 * area) * (x1 * y2 - y1 * x2 + (y1 - y2) * x + (x2 - x1) * y)
    return (s >= 0) & (t >= 0) & ((1 - s - t) >= 0)


# def get_maze(scaled_obstacles, maze_width, maze_height):
#     maze = [[0] * maze_width for _ in range(maze_height)]
#     for obstacle in scaled_obstacles:
#         for x in range(maze_width):
#             for y in range(maze_height):
#                 if is_point_in_triangle(x, y, obstacle):
#                     maze[x][y] = 1
#     return maze


def get_maze(scaled_obstacles, maze_width, maze_height, gap_scale):
    maze = np.zeros((maze_height, maze_width))
    for obstacle in scaled_obstacles:
        for x in range(maze_width):
            for y in range(maze_height):
                if is_point_in_triangle(x, y, obstacle, scale=gap_scale):
                    maze[x][y] = 1
    return maze


# def get_scaled_obstacles(obstacles, scale):
#     # Multiply each coordinate by scale
#     scaled_obstacles = []
#     for obstacle in obstacles:
#         scaled_triangle = [(scale * x, scale * y) for x, y in obstacle]
#         scaled_obstacles.append(scaled_triangle)
#     return scaled_obstacles


def get_scaled_obstacles(obstacles, scale):
    # Multiply each coordinate by scale
    scaled_obstacles = []
    for obstacle in obstacles:
        scaled_triangle = scale * obstacle
        scaled_obstacles.append(scaled_triangle)
    return scaled_obstacles


def main():
    # obstacles = [np.array([[1, 4], [2, 9], [7, 8]]), np.array([[6, 5], [6, 7], [8, 8]])]

    file_extension = '.npy'
    triangle_dir = 'out/whole_multi_trajectory/'
    index = 'T55'
    triangle_file = index + file_extension
    triangle_path = os.path.join(triangle_dir, triangle_file)
    triangle_mesh = np.load(triangle_path)
    triangle_mesh += 1
    start = (-1, -1)
    total_path = [start]
    time_step = 4
    scale = 50
    end = (2 * scale, 2 * scale)
    # end = (0, 2 * scale)
    for i in range(2, triangle_mesh.shape[1]):
        if i < triangle_mesh.shape[1] - 1:
            obstacles = []
            for j in range(16):
                obstacles.append(triangle_mesh[j][i])
                obstacles.append(triangle_mesh[j][i+1])
            # obstacles = [triangle_mesh[0][i], triangle_mesh[0][i+1], triangle_mesh[1][i], triangle_mesh[1][i+1], triangle_mesh[2][i], triangle_mesh[2][i+1], triangle_mesh[3][i], triangle_mesh[3][i+1]
            #              ]
        else:
            obstacles = []
            for k in range(16):
                obstacles.append(triangle_mesh[k][i])
            # obstacles = [triangle_mesh[0][i], triangle_mesh[1][i], triangle_mesh[2][i], triangle_mesh[3][i]]

        # Specify the resolution of the maze
        # maze_width = 1000
        # maze_height = 1000

        scaled_obstacles = get_scaled_obstacles(obstacles, scale)

        maze = get_maze(scaled_obstacles, 2*scale + 1, 2*scale + 1, gap_scale=4)
        # for row in maze:
        #     print([int(value) for value in row])

        # end = (10, 10)

        if i == 2:
            start = (scale * (start[0] + 1), scale * (start[1] + 1))
            # print(start)
        # print(start)
        path = astar(maze, start, end)
        if path is None:
            print("Encounter Collision")
            break
        real_path = [(x / scale - 1, y / scale - 1) for x, y in path]
        obstacles = [i - 1 for i in obstacles]

        if len(path) >= time_step:
            total_path += real_path[1:time_step]
            # print(total_path)
            start = path[time_step - 1]
        else:
            total_path += real_path[1:]
            start = path[-1]

        # print(path)

        if i % 20 == 0:
            # Plot the path and triangles
            plt.figure()
            plt.title('Path and Triangles Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')

            # Plot the path
            path_x, path_y = zip(*real_path)
            plt.plot(path_x, path_y, color='blue', marker='o', label='Path', linewidth=1, markersize=1)

            # Plot the triangles
            for obstacle in obstacles:
                x_coords, y_coords = obstacle.T
                plt.plot(np.append(x_coords, x_coords[0]), np.append(y_coords, y_coords[0]), color='red', marker='o',
                         )

            # Set the plot aspect ratio and legend
            plt.gca().set_aspect('equal')
            plt.legend()

            # Show the plot
            plt.show()

    print(len(total_path))
    save_dir = 'motion_planning/data/whole_astar_trajectory/'
    save_file = index + '_new' + file_extension
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_file)
    np.save(save_path, total_path)
    path_x, path_y = zip(*total_path)

    # Plot the path
    plt.figure()
    plt.title('Path')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.plot(path_x, path_y, color='blue', marker='o', label='Total Path', linewidth=1, markersize=1)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




