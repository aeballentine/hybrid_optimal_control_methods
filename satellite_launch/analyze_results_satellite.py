import numpy as np
import glob
import matplotlib.pyplot as plt

def read_text_file(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# overall logging
nn_time_log = []
nn_success_log = []
nn_cost_log = []
nn_final_H = []
nn_final_state = []

other_time_log = []
other_success_log = []
other_cost_log = []
other_final_H = []
other_final_state = []

for file in sorted(glob.glob('ref_path_transfer/rocket_launch_final.txt')):
    data = read_text_file(file)
    results_by_trial = data.split('~~~~~~~~~~~~~~~~~~~')

    # logging per-loop
    nn_time = []
    nn_success = []
    nn_cost = []
    nn_H = []
    nn_state = []

    other_time = []
    other_success = []
    other_cost = []
    other_H = []
    other_state = []

    for i, result in enumerate(results_by_trial):
        result = result.split('\n')
        result = [x for x in result if len(x) > 1]
        if len(result) < 1:
            continue
        trial_name, trial_time, trial_iterations, trial_guess, trial_sol, trial_residual, trial_state, = result

        final_accuracy = np.array([float(x) for x in trial_residual.split(':\t')[1].split('[')[1].split(']')[0].split(' ') if len(x) > 0])
        success = all(abs(final_accuracy) < 1e-4)
        cost = np.array([float(x) for x in trial_state.split(':\t')[1].split('[')[1].split(']')[0].split(' ') if len(x) > 0])[-1]
        solve_iterations = float(trial_iterations.split(': ')[1])

        initial_guess = np.array([float(x) for x in trial_guess.split(':\t')[1].split('[')[1].split(']')[0].split(' ') if len(x) > 0])
        solution = np.array([float(x) for x in trial_sol.split(':\t')[1].split('[')[1].split(']')[0].split(' ') if len(x) > 0])
        if 'NN' in trial_name:
            if len(nn_time) == 4:
                continue
            nn_time.append(solve_iterations)
            nn_success.append(success)
            nn_cost.append(cost)
            nn_H.append(abs(final_accuracy[3]))
            nn_state.append(final_accuracy)

        else:
            if len(other_time) == 10:
                continue
            other_time.append(solve_iterations)
            other_success.append(success)
            other_cost.append(cost)
            other_H.append(final_accuracy[3])
            other_state.append(final_accuracy)

    # if not any(nn_success):
    #     print(file)
    nn_time_log = np.array(nn_time)
    nn_success_log = np.array(nn_success)
    nn_cost_log = np.array(nn_cost)
    nn_final_H = np.array(nn_H)
    nn_final_state = np.array(nn_state)

    other_time_log = np.array(other_time)
    other_success_log = np.array(other_success)
    other_cost_log = np.array(other_cost)
    other_final_H = np.array(other_H)
    other_final_state = np.array(other_state)

print('Total trials: ', len(nn_time_log))
print('Method A')
print(f'Average iterations: {nn_time_log[0]:.1f}')
print(f'Number of successful trials: {100 * sum(nn_success_log) / len(nn_success_log):.1f}%')
print(f'Average residual: {np.sum(abs(nn_final_state))}')

print(' ')
print('Method B')
print(f'Number of successful trials: {100 * sum(other_success_log) / len(other_success_log):.1f}%')
print(f'Number of attempts (when successful): {np.mean(other_time_log[other_success_log]):.1f} +/- {np.std(other_time_log[other_success_log]):.1f}')
# print(f'Number of successful guesses (if any): {np.mean([sum(success) for success in other_success_log]):.1f}')
print(f'Average residual: {np.mean(abs(other_final_state[other_success_log]).sum(axis=1))}')
print(f'')

print(' ')