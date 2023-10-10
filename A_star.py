import math
import heapq
import uuid

import numpy as np


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0


def heuristic_distance(start, goal):
    return math.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)


def get_neighbors(node, obstacles):
    neighbors = []
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Assuming 4-connected grid

    for dx, dy in movements:
        new_x = node.position[0] + dx
        new_y = node.position[1] + dy

        if not is_collision((new_x, new_y), obstacles):
            neighbors.append(Node((new_x, new_y), node))

    return neighbors


def is_collision(position, obstacles):
    for obstacle in obstacles:
        if is_inside_triangle(position, obstacle):
            return True
    return False


def is_inside_triangle(position, triangle):
    A = tuple(triangle[0])
    B = tuple(triangle[1])
    C = tuple(triangle[2])
    P = position

    v0 = (C[0] - A[0], C[1] - A[1])
    v1 = (B[0] - A[0], B[1] - A[1])
    v2 = (P[0] - A[0], P[1] - A[1])

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= 0 and v >= 0 and u + v < 1


def a_star(start, goal, obstacles):
    open_set = []
    closed_set = set()

    start_node = Node(start)
    goal_node = Node(goal)

    heapq.heappush(open_set, (0, uuid.uuid4(), start_node))
    # print(open_set)

    while open_set:
        current_node = heapq.heappop(open_set)[2]
        # print(current_node)
        closed_set.add(current_node.position)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse the path to start from the beginning

        neighbors = get_neighbors(current_node, obstacles)
        for neighbor in neighbors:
            if neighbor.position in closed_set:
                continue

            neighbor.g_cost = current_node.g_cost + 1
            neighbor.h_cost = heuristic_distance(neighbor.position, goal)
            neighbor.f_cost = neighbor.g_cost + 0*neighbor.h_cost

            for i, (f, _, n) in enumerate(open_set):
                if neighbor == n and neighbor.f_cost > f:
                    break
            else:
                heapq.heappush(open_set, (neighbor.f_cost, uuid.uuid4(), neighbor))

    return None  # Path not found


# def is_collision(position, obstacles):
#     return np.any(np.isclose(obstacles, position, atol=1e-9))


# Example usage
triangle_obstacle = [(5, 6), (6, 7), (8, 4)]
start_point = (0, 0)

# print(is_collision(start_point, triangle_obstacle))
goal_point = (10, 10)

path = a_star(start_point, goal_point, [triangle_obstacle])

if path:
    print("Path found:")
    for position in path:
        print(position)
else:
    print("Path not found.")
#
