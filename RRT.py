import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None


class RRT:
    def __init__(self, start, goal, obstacles, step_size=0.1, max_iterations=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.nodes = []

    def rrt_algorithm(self):
        self.nodes.append(self.start)
        for _ in range(self.max_iterations):
            random_point = self.generate_random_point()
            nearest_node = self.find_nearest_node(random_point)
            new_point = self.extend(nearest_node, random_point)
            if new_point is not None and not self.check_collision(new_point):
                new_node = Node(new_point)
                new_node.parent = nearest_node
                self.nodes.append(new_node)
                if self.calculate_distance(new_node.point, self.goal.point) <= self.step_size:
                    self.goal.parent = new_node
                    return True
        return False

    def generate_random_point(self):
        return np.random.uniform(low=-10, high=10, size=2)

    def find_nearest_node(self, point):
        distances = [self.calculate_distance(node, point) for node in self.nodes]
        nearest_node = self.nodes[np.argmin(distances)]
        return nearest_node

    def extend(self, from_node, to_point):
        direction = self.calculate_direction(from_node.point, to_point)
        new_point = self.calculate_new_point(from_node.point, direction)
        return new_point

    def calculate_direction(self, from_point, to_point):
        direction = to_point - from_point
        norm = np.linalg.norm(direction)
        if norm > self.step_size:
            direction = (direction / norm) * self.step_size
        return direction

    def calculate_new_point(self, from_point, direction):
        new_point = from_point + direction
        return new_point

    def calculate_distance(self, node, point):
        return np.linalg.norm(node.point - point)

    def check_collision(self, point):
        for obstacle in self.obstacles:
            if self.point_in_triangle(point, obstacle):
                return True
        return False

    def point_in_triangle(self, point, triangle):
        a, b, c = triangle
        v0 = c - a
        v1 = b - a
        v2 = point - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return u >= 0 and v >= 0 and u + v <= 1

    def generate_trajectory(self, n):
        trajectory = []
        current_node = self.goal
        while current_node.parent:
            trajectory.append(current_node.point)
            current_node = current_node.parent
        trajectory.append(self.start.point)
        trajectory.reverse()
        trajectory = self.interpolate_trajectory(trajectory, n)
        return np.array(trajectory)

    def interpolate_trajectory(self, trajectory, n):
        interpolated_trajectory = []
        for i in range(len(trajectory) - 1):
            current_point = trajectory[i]
            next_point = trajectory[i + 1]
            direction = self.calculate_direction(current_point, next_point)
            num_steps = int(np.linalg.norm(direction) / self.step_size)
            interpolated_trajectory.extend(
                [self.calculate_new_point(current_point, direction)] * (num_steps * n)
            )
        return interpolated_trajectory

    def generate_rest_steps(self, goal, n, total_steps):
        rest_steps = [goal] * (total_steps - len(self.nodes) * n)
        return rest_steps

    def plan_trajectory(self, n):
        if self.rrt_algorithm():
            trajectory = self.generate_trajectory(n)
            rest_steps = self.generate_rest_steps(self.goal.point, n, total_steps=120)
            trajectory.extend(rest_steps)
            return trajectory
        else:
            raise Exception("RRT failed to find a feasible path.")

    def visualize_trajectory(self, trajectory):
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.scatter(self.start.point[0], self.start.point[1], color="green", label="Start")
        plt.scatter(self.goal.point[0], self.goal.point[1], color="red", label="Goal")
        for obstacle in self.obstacles:
            plt.fill(obstacle[:, 0], obstacle[:, 1], "b")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Agent Trajectory")
        plt.show()

    # Main code
def main():
    # Define start and goal points
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])

    # Define triangle mesh vertices for each time step
    triangle_mesh = np.load('out/trajectory/t1.npy')  # Assuming each triangle has 3 vertices

    # Set the radius and number of steps per time step
    radius = 0.1
    n = 1  # Adjust this value as needed

    # Create the RRT planner
    rrt_planner = RRT(start, goal, triangle_mesh)

    # Plan the trajectory
    trajectory = rrt_planner.plan_trajectory(n)

    # Visualize the trajectory
    rrt_planner.visualize_trajectory(trajectory)

if __name__ == "__main__":
    main()