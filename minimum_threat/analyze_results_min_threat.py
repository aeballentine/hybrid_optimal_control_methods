import numpy as np
import glob
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.figsize": (7, 8),  # Default figure size
    "axes.titlesize": 17,  # Title font size
    "axes.labelsize": 15,  # X and Y label size
    "xtick.labelsize": 10,  # X-axis tick labels
    "ytick.labelsize": 10,  # Y-axis tick labels
    "legend.fontsize": 12,  # Legend font size
    "lines.linewidth": 1.25,  # Thicker plot lines
    "lines.markersize": 6,  # Marker size
    "axes.grid": True,  # Enable grid
    "grid.alpha": 0.3,  # Transparent grid
    "grid.linestyle": "--",  # Dashed grid lines
    "axes.spines.top": False,  # Remove top spine
    "axes.spines.right": False,  # Remove right spine
    "legend.frameon": True,  # Remove legend box border
    "font.family": "Times New Roman",  # Use a clean, readable font
    "text.usetex": True
})


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

files = sorted(glob.glob('trial_three_shooting/*.txt'))
files =[file for file in files if file != 'trial_three_shooting/0.0_0.0_analytic_dynamic_random.txt']
for file in files:
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
        if 'NN' in result[0]:
            trial_name, initial_cost_guess, trial_time, trial_iterations, trial_guess, trial_sol, trial_residual, trial_state, = result
        else:
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
            # if not success:
            #     print(trial_guess)
            #     print(trial_residual)
            #     print(trial_sol)
            # else:
            #     print(abs(initial_guess - solution))
            # print(' ')
            nn_time.append(solve_iterations)
            nn_success.append(success)
            nn_cost.append(cost)
            nn_H.append(abs(final_accuracy[2]))
            nn_state.append((final_accuracy[0] ** 2 + final_accuracy[1] ** 2) ** 0.5)

        else:
            if len(other_time) == 10:
                continue
            other_time.append(solve_iterations)
            other_success.append(success)
            other_cost.append(cost)
            other_H.append(final_accuracy[2])
            other_state.append((final_accuracy[0] ** 2 + final_accuracy[1] ** 2) ** 0.5)

    # if not any(nn_success):
    #     print(file)
    nn_time_log.append(nn_time)
    nn_success_log.append(nn_success)
    nn_cost_log.append(nn_cost)
    nn_final_H.append(nn_H)
    nn_final_state.append(nn_state)

    other_time_log.append(other_time)
    other_success_log.append(other_success)
    other_cost_log.append(other_cost)
    other_final_H.append(other_H)
    other_final_state.append(other_state)

print('Total trials: ', len(nn_time_log))
print(f'Average iterations (hybrid): {np.mean([x for i, x in enumerate(sum(nn_time_log, [])) if sum(nn_success_log, [])[i]]):.1f}')
print(f'Success (hybrid): {100 * np.mean([any(success) for success in nn_success_log]):.1f}%')
print(f'Number of successful guesses (if any): {np.mean([sum(success) for success in nn_success_log]):.1f}')


print(' ')
# print(np.mean(other_time_log))
print(f'Trials with any number of successful trials (numerical): {100 * np.mean([any(success) for success in other_success_log]):.1f}%')
print(f'Average number of attempts (when successful): {np.mean([x for i, x in enumerate(sum(other_time_log, [])) if sum(other_success_log, [])[i]]):.1f}')
print(f'Number of successful guesses (if any): {np.mean([sum(success) for success in other_success_log]):.1f}')

print(' ')
print([sum(success) for success in nn_success_log])
print([sum(success) for success in other_success_log])

print(' ')
for i, x in enumerate(nn_cost_log):
    # if any(nn_success_log[i]):
    print(f'Ratio of attempts (numerical / hybrid): {np.mean([x for j, x in enumerate(other_time_log[i]) if other_success_log[i][j]]) / np.mean([iterations for j, iterations in enumerate(nn_time_log[i]) if nn_success_log[i][j]]):.1f}')
    print(f'Hybrid number of iterations: {[iterations for j, iterations in enumerate(nn_time_log[i]) if nn_success_log[i][j]]}')
    print(f'Hybrid cost: {[float(val) for j, val in enumerate(x) if nn_success_log[i][j]]}')
    print(f'Number numerical paths: {sum(other_success_log[i])}')
    if any(other_success_log[i]):
        print(f'Min numerical cost: {min([float(y) for j, y in enumerate(other_cost_log[i]) if other_success_log[i][j]])}')
        print(f'Max numerical cost: {max([float(y) for j, y in enumerate(other_cost_log[i]) if other_success_log[i][j]])}')
    print('`````')
print('Done')

nn_final_H = np.array(nn_final_H)
nn_final_state = np.array(nn_final_state)
nn_success_log = np.array(nn_success_log)
nn_time_log = np.array(nn_time_log)
nn_cost_log = np.array(nn_cost_log)

nn_state = np.zeros(48)
nn_H = np.zeros(48)
nn_cost = np.zeros(48)
nn_iterations = np.zeros(48)
for i in range(48):
    if any(nn_success_log[i]):
        nn_c = nn_cost_log[i][nn_success_log[i]]
        nn_it = nn_time_log[i][nn_success_log[i]]
        state = nn_final_state[i][nn_success_log[i]]
        hamiltonian = nn_final_H[i][nn_success_log[i]]
        ind = np.where(nn_c == nn_c.min())[0]

        nn_state[i] = state[ind].mean()
        nn_iterations[i] = nn_it[ind].mean()
        nn_H[i] = hamiltonian[ind].mean()
        nn_cost[i] = nn_c.min()
    elif any(nn_final_state[i] < 1e-1):
        success = nn_final_state[i] < 1e-1
        nn_c = nn_cost_log[i][success]
        nn_it = nn_time_log[i][success]
        state = nn_final_state[i][success]
        hamiltonian = nn_final_H[i][success]

        ind = np.where((state+hamiltonian) == (state+hamiltonian).min())[0]
        ind_2 = np.where(nn_c[ind] == nn_c[ind].min())[0]

        nn_state[i] = state[ind][ind_2].mean()
        nn_iterations[i] = nn_it[ind][ind_2].mean()
        nn_H[i] = hamiltonian[ind][ind_2].mean()
        nn_cost[i] = nn_c[ind][ind_2].mean()
    else:
        print('No valid paths')

other_final_H = np.array(other_final_H)
other_final_state = np.array(other_final_state)
other_success_log = np.array(other_success_log)
other_time_log = np.array(other_time_log)
other_cost_log = np.array(other_cost_log)

other_state = np.zeros(48)
other_H = np.zeros(48)
other_cost = np.zeros(48)
other_iterations = np.zeros(48)
for i in range(len(other_success_log)):
    print(other_success_log[i])
    if any(other_success_log[i]):
        other_c = other_cost_log[i][other_success_log[i]]
        other_it = other_time_log[i][other_success_log[i]]
        state = other_final_state[i][other_success_log[i]]
        hamiltonian = other_final_H[i][other_success_log[i]]
        ind = np.where(other_c == other_c.min())[0]

        other_state[i] = state[ind].mean()
        other_iterations[i] = other_it[ind].mean()
        other_H[i] = hamiltonian[ind].mean()
        other_cost[i] = other_c.min()
    elif any(other_final_state[i] < 1e-1):
        success = other_final_state[i] < 1e-1
        other_c = other_cost_log[i][success]
        other_it = other_time_log[i][success]
        state = other_final_state[i][success]
        hamiltonian = other_final_H[i][success]

        ind = np.where((state+hamiltonian) == (state+hamiltonian).min())[0]
        ind_2 = np.where(other_c[ind] == other_c[ind].min())[0]
        other_state[i] = state[ind][ind_2].mean()
        other_iterations[i] = other_it[ind][ind_2].mean()
        other_H[i] = hamiltonian[ind][ind_2].mean()
        other_cost[i] = other_c[ind][ind_2].mean()
    else:
        print('No valid paths')

# valid = nn_valid & other_valid  # only compare cases valid for both solvers
valid = np.ones(48, dtype=np.bool)
print(np.mean(other_cost / nn_cost))
metric_names = ['Cost', 'Iterations', '$\mathcal{L}1$-norm']
metric_ratios = np.array([
    other_cost / nn_cost,
    other_iterations / nn_iterations,
    (other_state + np.abs(other_H)) / (nn_state + np.abs(nn_H)),
])
print(np.where(metric_ratios[0] < 1e-3))
ind = np.where(metric_ratios[0] < 1e-3)[0]
print(np.array(files)[ind])

print(np.mean(metric_ratios, axis=1))
print(10 ** np.mean(np.log10(metric_ratios), axis=1))
print(np.median(metric_ratios, axis=1))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.ravel()

for ax, name, ratio in zip(axes, metric_names, metric_ratios):
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    log_ratio = np.log10(ratio)

    bound = max(0.5, np.ceil(np.abs(log_ratio).max() * 10) / 10)
    bins = np.linspace(-bound, bound, 41)

    ax.hist(log_ratio, bins=bins, color='#4C72B0', edgecolor='white', alpha=0.9)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.0, label='Equal performance')

    median_log = np.median(log_ratio)
    ax.axvline(median_log, color='#DD8452', linewidth=1.3,
               label=f'Median = {10 ** median_log:.2f}')
    # mean_log = np.mean(ratio)
    # ax.axvline(np.log10(mean_log), color='#55A868', linewidth=1.3,
    #            label=f'Mean = {mean_log:.2e}')

    pct_better = (log_ratio > 0).mean() * 100
    pct_worse = (log_ratio < 0).mean() * 100
    name_bold = name.replace(' ', r'\ ')
    ax.set_title(name, fontweight='bold', y=1.12)
    ax.text(0.5, 1.02, f'Hybrid method improves performance: {pct_better:.0f}\%\nNumerical method performs better: {pct_worse:.0f}\%', ha='center', va='bottom',
            transform=ax.transAxes, fontstyle='italic')
    # ax.set_title(rf'$\bf{{{name}}}$' + f'\nHybrid method improve performance: {pct_better:.0f}%\nNumerical method performs better: {pct_worse:.0f}%',
    #              fontsize=11)
    # ax.set_title(f'{name}\nNN better in {pct_better:.0f}% of cases (n={len(log_ratio)})',
    #              fontsize=11, fontweight='bold')
    # ax.set_xlabel(rf'$\leftarrow$ numerical outperforms hybrid         hybrid outperforms numerical $\rightarrow$   ')
    ax.set_xlabel(r'$\log_{10}\left(\frac{\mathrm{numerical}}{\mathrm{hybrid}}\right)$')
    ax.set_ylabel('Number of cases')
    ax.legend(loc='upper left')
    # ax.grid(alpha=0.3)

# fig.suptitle('Comparison of Hybrid and Numerical Methods',
#              fontsize=14, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('figures/comparison.eps', dpi=600, format='eps', bbox_inches='tight')

print(f"{'Metric':<22}{'NN wins':>10}{'Mean factor':>16}")
print('-' * 48)
for name, ratio in zip(metric_names, metric_ratios):
    # r = ratio[np.isfinite(ratio) & (ratio > 0)]
    r = ratio
    pct = (r > 1).mean() * 100
    median = np.mean(r)
    print(f'{name:<22}{pct:>9.0f}%{median:>14.2e}\u00d7')

plt.show()