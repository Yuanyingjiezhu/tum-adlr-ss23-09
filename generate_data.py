from BPS import BPS

x_range = (-1, 1)
y_range = (-1, 1)
num_train_data = 10000
num_eval_data = 1000
num_test_data = 1000
num_points = 50
Generator = BPS(num_train_data, num_eval_data, num_test_data, x_range, y_range, num_points)
Generator.cal_sdf()
