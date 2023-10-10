


test_state = sequential_data[0]
test_agent_position = np.array([0, 0]) # Agent's initial position
while True:
if len(test_state.shape) == 1:
test_state = np.expand_dims(test_state, axis=0)
state = np.concatenate((test_state[-STATE_HISTORY:].flatten(), test_agent_position, start_point, goal_point))
action, _ = agent.select_action(state)
if action == 0: # Move up
test_agent_position[1] += 1
elif action == 1: # Move down
test_agent_position[1] -= 1
elif action == 2: # Move left
test_agent_position[0] -= 1
elif action == 3: # Move right
test_agent_position[0] += 1