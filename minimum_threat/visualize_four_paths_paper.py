import matplotlib.pyplot as plt
from shooting_min_threat import threat
import pandas as pd
import numpy as np


plt.rcParams.update({
    "figure.figsize": (7, 8),  # Default figure size
    "axes.titlesize": 15,  # Title font size
    "axes.labelsize": 12,  # X and Y label size
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


paths = pd.read_csv('trial_three/-15.0_-15.0_NN_path.csv')
t = paths.t.to_numpy()
t_zeros = np.where(t == 0)[0]
max_time = max(t[t_zeros - 1])
t_intervals = max_time / 4
path_x = []
path_y = []
path_t = []
for i in range(4):
    path_x.append(paths.x1.to_numpy()[t_zeros[i]:t_zeros[(i+1)%4]-1])
    path_y.append(paths.x2.to_numpy()[t_zeros[i]:t_zeros[(i+1)%4]-1])
    path_t.append(paths.t.to_numpy()[t_zeros[i]:t_zeros[(i+1)%4]-1])

x = np.linspace(-20, 20, 200)
y = np.linspace(-20, 20, 200)
X, Y = np.meshgrid(x, y, indexing="xy")
fig = plt.figure(figsize=(12, 3))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1)
    Z, _, _ = threat(X, Y, t_intervals * (i + 1))
    im = ax.imshow(
        Z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        animated=True,
        zorder=1
    )
    for j in range(4):
        plt.plot(path_x[j][path_t[i] < t_intervals * (i + 1)], path_y[j][path_t[i] < t_intervals * (i + 1)],
                 color='red', linestyle='dashed')
    plt.title(f't={t_intervals * (i + 1):.1f}s')
    plt.xlabel(r'$x_1$ [m]')
    plt.ylabel(r'$x_2$ [m]')

plt.tight_layout()
plt.savefig("figures/dynamic_traj.eps", format='eps', dpi=600, bbox_inches="tight")
# plt.show()