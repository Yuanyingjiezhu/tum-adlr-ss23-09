import math

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# def is_collision(point, obstacle):
#     x, y = point
#     x1, y1 = obstacle[0]
#     x2, y2 = obstacle[1]
#     x3, y3 = obstacle[2]
#
#     # Calculate the area of the triangle formed by the point and the obstacle vertices
#     area = abs(0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
#
#     # Check if the point is inside the triangle (collision)
#     if area > 0 and area <= 0.5:
#         return True
#
#     return False


# def is_collision(position, obstacles):
#     for obstacle in obstacles:
#         if is_inside_triangle(position, obstacle):
#             return True
#     return False


def is_collision(position, triangle):
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


def get_neighbors(point, obstacles, step_size):
    neighbors = []
    x, y = point

    # Generate neighboring points in a square region around the current point
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            new_x = x + dx
            new_y = y + dy

            # Check if the new point is within the valid range
            if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                new_point = (new_x, new_y)

                # Check for collisions with obstacles
                collision = False
                for obstacle in obstacles:
                    if is_collision(new_point, obstacle):
                        collision = True
                        break

                if not collision:
                    neighbors.append(new_point)

    return neighbors

def a_star(start_point, goal_point, obstacles):
    # Create start and goal nodes
    start_node = Node(start_point)
    goal_node = Node(goal_point)

    # Initialize open and closed lists
    open_list = []
    closed_list = []

    # Add the start node to the open list
    open_list.append(start_node)

    # Loop until the goal node is found or the open list is empty
    while open_list:
        # Get the node with the lowest f score from the open list
        current_node = min(open_list, key=lambda node: node.f)

        # Move the current node from the open list to the closed list
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Check if the goal node has been reached
        if current_node.position == goal_node.position:
            path = []
            node = current_node

            # Reconstruct the path from the goal node to the start node
            while node is not None:
                path.append(node.position)
                node = node.parent

            # Return the path in reverse order
            return path[::-1]

        # Generate neighboring points
        neighbors = get_neighbors(current_node.position, obstacles, step_size=0.1)

        for neighbor_position in neighbors:
            # Create a new node for the neighbor
            neighbor_node = Node(neighbor_position, parent=current_node)

            # Check if the neighbor is already in the closed list
            if neighbor_node in closed_list:
                continue

            # Calculate g, h, and f scores for the neighbor
            neighbor_node.g = current_node.g + distance(current_node.position, neighbor_node.position)
            neighbor_node.h = distance(neighbor_node.position, goal_node.position)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            # Check if the neighbor is already in the open list and has a lower f score
            if neighbor_node in open_list and neighbor_node.g > current_node.g:
                continue

            # Add the neighbor to the open list
            open_list.append(neighbor_node)

    # No path found
    return None

# Example usage
triangle_obstacle = [(0.3, 0.5), (0.6, 0.7), (0.8, 0.4)]
start_point = (0, 0)
goal_point = (1, 1)

path = a_star(start_point, goal_point, [triangle_obstacle])
print("Path:", path)