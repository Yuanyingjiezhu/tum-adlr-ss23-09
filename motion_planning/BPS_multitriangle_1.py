import numpy as np
import matplotlib.pyplot as plt


def find_nearest_point(point, line_start, line_end):
    # Calculate the direction vector of the line segment
    line_vec = line_end - line_start

    # Find the vector from line_start to point_to_check
    point_vec = point - line_start

    # Calculate the projection of point_vec onto the line_vec
    projection = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)

    # Calculate the foot of the perpendicular
    foot_of_perpendicular = line_start + projection * line_vec

    # Check if the foot lies within the line segment
    if 0 <= projection <= 1:
        return foot_of_perpendicular
    else:
        # Find the nearest endpoint of the line segment
        distance_to_start = np.linalg.norm(point - line_start)
        distance_to_end = np.linalg.norm(point - line_end)

        if distance_to_start < distance_to_end:
            return line_start
        else:
            return line_end


def point_in_polygon(point, dynamic_points):
    for polygon in dynamic_points:
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
        if num_intersections % 2 == 1:
            return True
    return False


def find_surrounding_edges(observed_points):
    multi_triangle = []
    for triangle in observed_points:
        triangle.reshape((3, 2))
        surrounding_edges = np.zeros((3, 2, 2))
        for i in range(2):
            surrounding_edges[i, 0] = triangle[i]
            surrounding_edges[i, 1] = triangle[i + 1]
        surrounding_edges[2, 0] = triangle[2]
        surrounding_edges[2, 1] = triangle[0]
        multi_triangle.append(surrounding_edges)
    multi_triangle = np.array(multi_triangle)
    return multi_triangle


class BPS:
    def __init__(self,  x_range, y_range, num_points, dynamic_pointset, save_path, num_objects):
        self.x_range = x_range
        self.y_range = y_range
        self.num_points = num_points
        self.dynamic_pointset = dynamic_pointset
        self.save_path = save_path

    def build_base_points(self):
        base_points = []
        x_coords = np.linspace(-1, 1, self.num_points)
        y_coords = np.linspace(-1, 1, self.num_points)
        X, Y = np.meshgrid(x_coords, y_coords)
        x_samples = X.flatten()
        y_samples = Y.flatten()
        for i in range(len(x_samples)):
            point = (x_samples[i], y_samples[i])
            base_points.append(point)
        return base_points

    def cal_sdf(self):
        dynamic_pointset = np.array(self.dynamic_pointset).transpose((1, 0, 2, 3))
        with open(save_path, 'w') as file:
            pass
        for dynamic_points in dynamic_pointset:
            dynamic_points = dynamic_points.reshape(num_objects, 3, 2)
            for triangle in dynamic_points:
                x_values = [triangle[i][0] for i in range(3)] + [triangle[0][0]]
                y_values = [triangle[i][1] for i in range(3)] + [triangle[0][1]]
                plt.plot(x_values, y_values, 'b-', label='convex hull')
            multi_triangle = find_surrounding_edges(dynamic_points)
            all_edges = multi_triangle.reshape(3*num_objects, -1)
            all_edges_endpoints = all_edges.reshape((3*num_objects, 2, 2))
            base_points = np.array(self.build_base_points())
            x = [point[0] for point in base_points]
            y = [point[1] for point in base_points]
            plt.scatter(x, y, c='r', label='base points')
            min_distance = np.array([1000.0] * len(base_points))
            sdf = np.array([1000.0] * len(base_points))
            min_point = np.array([(0.0, 0.0)] * len(base_points))
            for i in range(len(base_points)):
                for j in range(len(all_edges)):
                    nearest_point = find_nearest_point(base_points[i], all_edges_endpoints[j, 0], all_edges_endpoints[j, 1])
                    distance = np.linalg.norm(base_points[i] - nearest_point)
                    if min_distance[i] > distance:
                        min_distance[i] = distance
                        min_point[i] = nearest_point
                        sdf[i] = distance
                        if point_in_polygon(base_points[i], dynamic_points):
                            sdf[i] = -min_distance[i]
            with open(save_path, 'a') as file:
                row_str = ' '.join([str(element) for element in sdf])
                file.write(row_str + '\n')

            # Plotting the nearest point
            plt.plot(min_point[:, 0], min_point[:, 1], 'go', label='nearest points')
            # Plotting the shortest distance
            plt.plot([base_points[:, 0], min_point[:, 0]], [base_points[:, 1], min_point[:, 1]], 'r--')

            # Add labels and legend
            line_for_convex = plt.Line2D([], [], color='blue', linestyle='-', label='convex hull')
            line_for_distance = plt.Line2D([], [], color='red', linestyle='--', label='Shortest Distance')
            point_for_observe = plt.Line2D([], [], color='blue', marker='o', linestyle='None', label='observed points')
            point_for_base = plt.Line2D([], [], color='red', marker='o', linestyle='None', label='base points')
            point_for_nearest = plt.Line2D([], [], color='green', marker='o', linestyle='None', label='nearest points')

            # plot
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend(handles=[line_for_convex, line_for_distance, point_for_observe, point_for_base, point_for_nearest])
            # Set the aspect ratio of the plot
            plt.axis('equal')
            # Show the plot
            plt.show()
            plt.close()


num_basispoints = 50
num_objects = 16
dynamic_pointset = []
x_range = (-1, 1)
y_range = (-1, 1)
save_path = 'data/second_trajectory'
first_trajectory = np.load('data/T01.npy')
Generator = BPS(x_range, y_range, num_basispoints, first_trajectory, save_path, num_objects)
Generator.cal_sdf()
