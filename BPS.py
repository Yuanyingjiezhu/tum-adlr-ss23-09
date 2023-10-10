import numpy as np
import random
import matplotlib.pyplot as plt
import torch


# x_range = (-1, 1)
# y_range = (-1, 1)
# num_train_data = 10
# num_test_data = 4
# num_points = 5


def find_nearest_point(point, line_start, line_end):
    # Calculate the vector components
    line_vec = line_end - line_start
    point_vec = point - line_start

    # Calculate the projection of the point vector onto the line vector
    projection = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)

    # Clamp the projection within the line segment
    projection = np.clip(projection, 0, np.linalg.norm(line_vec))

    # Calculate the closest point on the line segment
    closest_point = line_start + (projection / np.linalg.norm(line_vec)) * line_vec

    return closest_point


def point_in_polygon(point, polygon):
    num_intersections = 0
    x, y = point

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        if y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1]) and x <= max(p1[0], p2[0]):
            if p1[1] != p2[1]:
                intersection_x = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or x <= intersection_x:
                    num_intersections += 1

    return num_intersections % 2 == 1


def find_surrounding_edges(observed_points):
    surrounding_edges = np.array([[[0.0, 0.0] for _ in range(2)] for _ in range(len(observed_points))])
    for i in range(len(observed_points) - 1):
        surrounding_edges[i, 0] = observed_points[i]
        surrounding_edges[i, 1] = observed_points[i + 1]
    surrounding_edges[len(observed_points) - 1, 0] = observed_points[len(observed_points) - 1]
    surrounding_edges[len(observed_points) - 1, 1] = observed_points[0]

    # for i, edge in enumerate(surrounding_edges):
    #     point1, point2 = edge
    # print(f"Edge {i + 1}: {point1} to {point2}")

    # Plot the edges
    for edge in surrounding_edges:
        x_values = [edge[0][0], edge[1][0]]
        y_values = [edge[0][1], edge[1][1]]
        plt.plot(x_values, y_values, 'b-')
    return surrounding_edges


class BPS:
    def __init__(self, num_train_data, num_eval_data, num_test_data, x_range, y_range, num_points):
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data
        self.num_test_data = num_test_data
        self.x_range = x_range
        self.y_range = y_range
        self.num_points = num_points

    def build_base_points(self):
        base_points = []
        # Define the number of points and the area of the square
        num_base_points = self.num_points * self.num_points
        # Generate the points
        x_coords = np.linspace(-1, 1, self.num_points)
        y_coords = np.linspace(-1, 1, self.num_points)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Flatten the meshgrid to obtain the individual coordinates
        x_samples = X.flatten()
        y_samples = Y.flatten()
        for i in range(len(x_samples)):
            point = (x_samples[i], y_samples[i])
            base_points.append(point)
        return base_points

    def cal_sdf(self):
        # sample n set of 3 observed points
        observed_points_set = []
        eval_observed_points_set = []
        test_observed_points_set = []

        for i in range(self.num_train_data):
            for j in range(3):
                x = random.uniform(self.x_range[0], self.x_range[1])
                y = random.uniform(self.y_range[0], self.y_range[1])
                observed_points_set.append((x, y))
        for i in range(self.num_eval_data):
            for j in range(3):
                x = random.uniform(self.x_range[0], self.x_range[1])
                y = random.uniform(self.y_range[0], self.y_range[1])
                eval_observed_points_set.append((x, y))
        for i in range(self.num_test_data):
            for j in range(3):
                x = random.uniform(self.x_range[0], self.x_range[1])
                y = random.uniform(self.y_range[0], self.y_range[1])
                test_observed_points_set.append((x, y))
        #
        observed_points_set = np.array(observed_points_set).reshape((self.num_train_data, 3, 2))
        # determinant = np.linalg.det(np.stack([observed_points_set[:, 1] - observed_points_set[:, 0], observed_points_set[:, 2] - observed_points_set[:, 0]]))
        # indices_ccw = np.where(determinant < 0)
        # observed_points_set[indices_ccw] = np.flip(observed_points_set[indices_ccw], axis=1)
        # print("aaaa:", type(observed_points_set))
        observed_points_flat = observed_points_set.reshape(-1, 6)
        with open('data/observed_points_dataset', 'w') as file:
            for row in observed_points_flat:
                file.write(' '.join([str(elem) for elem in row]) + '\n')
        eval_observed_points_set = np.array(eval_observed_points_set).reshape((self.num_eval_data, 3, 2))
        eval_observed_points_flat = eval_observed_points_set.reshape(-1, 6)
        with open('data/eval_observed_points_dataset', 'w') as file:
            for row in eval_observed_points_flat:
                file.write(' '.join([str(elem) for elem in row]) + '\n')
        test_observed_points_set = np.array(test_observed_points_set).reshape((self.num_test_data, 3, 2))
        test_observed_points_flat = test_observed_points_set.reshape(-1, 6)
        with open('data/test_observed_points_dataset', 'w') as file:
            for row in test_observed_points_flat:
                file.write(' '.join([str(elem) for elem in row]) + '\n')
        with open('data/sdf_dataset', 'w') as file:
            pass
        for observed_points in observed_points_set:
            observed_points = observed_points.reshape(3, 2)

            # Create a scatter plot of the observed points
            # x = [point[0] for point in observed_points]
            # y = [point[1] for point in observed_points]
            # plt.scatter(x, y, c='b', label='observed points')

            # Find surrounding edges
            surrounding_edges = find_surrounding_edges(observed_points)

            # compute and plot base points
            base_points = np.array(self.build_base_points())
            # x = [point[0] for point in base_points]
            # y = [point[1] for point in base_points]
            # plt.scatter(x, y, c='r', label='base points')

            # compute the nearest point and shortest distance
            min_distance = np.array([1000.0] * len(base_points))
            sdf = np.array([1000.0] * len(base_points))
            min_point = np.array([(0.0, 0.0)] * len(base_points))
            for i in range(len(base_points)):
                for j in range(len(surrounding_edges)):
                    nearest_point = find_nearest_point(base_points[i], surrounding_edges[j, 0], surrounding_edges[j, 1])
                    distance = np.linalg.norm(base_points[i] - nearest_point)
                    if min_distance[i] > distance:
                        min_distance[i] = distance
                        min_point[i] = nearest_point
                        sdf[i] = distance
                        if point_in_polygon(base_points[i], observed_points):
                            sdf[i] = -min_distance[i]

                # print(f"The nearest point is: {min_point[i]}")
                # print(f"The nearest distance is: {min_distance[i]}")

            # Plotting the nearest point
            # plt.plot(min_point[:, 0], min_point[:, 1], 'go', label='nearest points')
            # Plotting the shortest distance
            # plt.plot([base_points[:, 0], min_point[:, 0]], [base_points[:, 1], min_point[:, 1]], 'r--')

            # Add labels and legend
            # line_for_convex = plt.Line2D([], [], color='blue', linestyle='-', label='convex hull')
            # line_for_distance = plt.Line2D([], [], color='red', linestyle='--', label='Shortest Distance')
            # point_for_observe = plt.Line2D([], [], color='blue', marker='o', linestyle='None', label='observed points')
            # point_for_base = plt.Line2D([], [], color='red', marker='o', linestyle='None', label='base points')
            # point_for_nearest = plt.Line2D([], [], color='green', marker='o', linestyle='None', label='nearest points')

            # plot
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.legend(
            #     handles=[line_for_convex, line_for_distance, point_for_observe, point_for_base, point_for_nearest])
            # Set the aspect ratio of the plot
            # plt.axis('equal')
            # Show the plot
            # plt.show()
            # plt.close()
            print(sdf)
            # Write sdf to file at each iteration without overwriting
            with open('data/sdf_dataset', 'a') as file:
                row_str = ' '.join([str(element) for element in sdf])  # Convert each element to string
                file.write(row_str + '\n')

        with open('data/test_sdf_dataset', 'w') as file:
            pass
        for observed_points in test_observed_points_set:
            observed_points = observed_points.reshape(3, 2)

            # Create a scatter plot of the observed points
            x = [point[0] for point in observed_points]
            y = [point[1] for point in observed_points]

            # Find surrounding edges
            surrounding_edges = find_surrounding_edges(observed_points)

            # compute and plot base points
            base_points = np.array(self.build_base_points())
            x = [point[0] for point in base_points]
            y = [point[1] for point in base_points]

            # compute the nearest point and shortest distance
            min_distance = np.array([1000.0] * len(base_points))
            sdf = np.array([1000.0] * len(base_points))
            min_point = np.array([(0.0, 0.0)] * len(base_points))
            for i in range(len(base_points)):
                for j in range(len(surrounding_edges)):
                    nearest_point = find_nearest_point(base_points[i], surrounding_edges[j, 0], surrounding_edges[j, 1])
                    distance = np.linalg.norm(base_points[i] - nearest_point)
                    if min_distance[i] > distance:
                        min_distance[i] = distance
                        min_point[i] = nearest_point
                        sdf[i] = distance
                        if point_in_polygon(base_points[i], observed_points):
                            sdf[i] = -min_distance[i]

                # print(f"The nearest point is: {min_point[i]}")
                # print(f"The nearest distance is: {min_distance[i]}")

            # Write sdf to file at each iteration without overwriting
            with open('data/test_sdf_dataset', 'a') as file:
                row_str = ' '.join([str(element) for element in sdf])  # Convert each element to string
                file.write(row_str + '\n')

        with open('data/eval_sdf_dataset', 'w') as file:
            pass
        for observed_points in eval_observed_points_set:
            observed_points = observed_points.reshape(3, 2)

            # Create a scatter plot of the observed points
            x = [point[0] for point in observed_points]
            y = [point[1] for point in observed_points]

            # Find surrounding edges
            surrounding_edges = find_surrounding_edges(observed_points)

            # compute and plot base points
            base_points = np.array(self.build_base_points())
            x = [point[0] for point in base_points]
            y = [point[1] for point in base_points]

            # compute the nearest point and shortest distance
            min_distance = np.array([1000.0] * len(base_points))
            sdf = np.array([1000.0] * len(base_points))
            min_point = np.array([(0.0, 0.0)] * len(base_points))
            for i in range(len(base_points)):
                for j in range(len(surrounding_edges)):
                    nearest_point = find_nearest_point(base_points[i], surrounding_edges[j, 0], surrounding_edges[j, 1])
                    distance = np.linalg.norm(base_points[i] - nearest_point)
                    if min_distance[i] > distance:
                        min_distance[i] = distance
                        min_point[i] = nearest_point
                        sdf[i] = distance
                        if point_in_polygon(base_points[i], observed_points):
                            sdf[i] = -min_distance[i]

                # print(f"The nearest point is: {min_point[i]}")
                # print(f"The nearest distance is: {min_distance[i]}")

            # Write sdf to file at each iteration without overwriting
            with open('data/eval_sdf_dataset', 'a') as file:
                row_str = ' '.join([str(element) for element in sdf])  # Convert each element to string
                file.write(row_str + '\n')

