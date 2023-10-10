# sample 10 base points and Print the sampled points
    # for _ in range(10):
    #     x = random.uniform(x_range[0], x_range[1])
    #     y = random.uniform(y_range[0], y_range[1])
    #     base_points.append((x, y))
    # for point in base_points:
    #     print("the base points are:",point)




# training_losses = []
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = src(inputs)
#         loss = criterion(outputs, targets)
#         training_losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
# plt.plot(training_losses)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()
# Initializing the list for storing the loss and accuracy


# def find_convexhull(points):
#     # Calculate the convex hull of the points
#     hull = ConvexHull(points)
#
#     # Get the indices of the vertices of the convex hull
#     hull_vertices = hull.vertices
#
#     # Get the coordinates of the vertices of the convex hull
#     hull_points = [points[i] for i in hull_vertices]
#
#     # Calculate the list of edges by connecting consecutive vertices of the convex hull
#     edges = [(hull_points[i], hull_points[(i + 1) % len(hull_points)]) for i in range(len(hull_points))]
#
#     return edges


# Calculate the displacement vectors between time steps i and i-1, i-2
displacement1 = triangle_mesh[i] - triangle_mesh[i-1]
displacement2 = triangle_mesh[i] - triangle_mesh[i-2]

# Calculate the dot product between the displacement vectors and triangle vertices
dot_product1 = np.sum(displacement1 * triangle_mesh[i], axis=1)
dot_product2 = np.sum(displacement2 * triangle_mesh[i], axis=1)

# Find the vertex at the front end of the movement direction
end_vertex_index = np.argmax(np.maximum(dot_product1, dot_product2))

# Find the indices of the other two vertices
other_vertex_indices = [i for i in range(3) if i != end_vertex_index]

# Calculate the predicted vertex coordinates at time step i+1 using velocity and acceleration
# velocity = displacement1[end_vertex_index] - displacement2[end_vertex_index]
# acceleration = displacement1[end_vertex_index] + displacement2[end_vertex_index] - 2 * triangle_mesh[i][end_vertex_index]

predicted_vertex = predict_next_point(triangle_mesh[i-2][end_vertex_index], triangle_mesh[i-1][end_vertex_index], triangle_mesh[i][end_vertex_index])

# Create the triangle using the predicted vertex and the other two vertices
triangle = np.vstack((predicted_vertex, triangle_mesh[i][other_vertex_indices[0]], triangle_mesh[i][other_vertex_indices[1]]))

# Plot the triangles
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# Plot the triangle at time step i
ax.plot(triangle_mesh[i][:, 0], triangle_mesh[i][:, 1], color='blue', marker='o', label='Triangle at Time Step i')

# Plot the triangle at time step i-1
ax.plot(triangle_mesh[i-1][:, 0], triangle_mesh[i-1][:, 1], color='green', marker='o', label='Triangle at Time Step i-1')

# Plot the triangle at time step i-2
ax.plot(triangle_mesh[i-2][:, 0], triangle_mesh[i-2][:, 1], color='red', marker='o', label='Triangle at Time Step i-2')

# Plot the newly formed triangle at time step i+1
ax.plot(triangle[:, 0], triangle[:, 1], color='orange', marker='o', label='New Triangle at Time Step i+1')

# Set plot title and labels
ax.set_title('Triangles at Different Time Steps')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Set the legend
ax.legend()

# Show the plot
plt.show()