import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

T = 2.093217
S = 218.1790e-12
rho = 61.05110e9
g = 1.144258
m0 = 1
mdot = 0.6
Cd = 0.5
h0 = 46.88889e-3

def hamiltonian(df, p3, p4, psi):
    D = 0.5 * rho * np.exp(-df.x2 / h0) * (df.x3 ** 2 + df.x4 ** 2) * S * Cd
    cos_gamma = df.x3 / (df.x3 ** 2 + df.x4 ** 2 + 1e-6) ** 0.5
    sin_gamma = df.x4 / (df.x3 ** 2 + df.x4 ** 2 + 1e-6) ** 0.5
    m = m0 - mdot * df.t
    H = 1 + df.p2 * df.x4 + p3 * (T * np.cos(psi) - D * cos_gamma) / m + p4 * ((T * np.sin(psi) - D * sin_gamma) / m - g)
    return H


data_nn = pd.read_csv('ref_path_transfer/rocket_launch_nn_solution_3.csv')
data_numeric = pd.read_csv('ref_path_transfer/rocket_launch_hybrid_solution.csv')

fig = plt.figure()
fig.add_subplot(2, 4, 1)
plt.plot(data_nn.t, data_nn.x1, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.x1, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 2)
plt.plot(data_nn.t, data_nn.x2, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.x2, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 3)
plt.plot(data_nn.t, data_nn.x3, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.x3, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 4)
plt.plot(data_nn.t, data_nn.x4, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.x4, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 5)
plt.plot(data_nn.t, data_nn.p2, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.p2, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 6)
plt.plot(data_nn.t, data_nn.p3, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.p3, label='Numeric', color='black', linestyle='dashdot')

fig.add_subplot(2, 4, 7)
plt.plot(data_nn.t, data_nn.p4, label='NN', color='red')
plt.plot(data_numeric.t, data_numeric.p4, label='Numeric', color='black', linestyle='dashdot')

u1 = np.arctan2(data_numeric.p4.to_numpy(), data_numeric.p3.to_numpy())
u2 = u1 + np.pi
u_vals = np.vstack([u1 % (2 * np.pi), u2 % (2 * np.pi)])
P_vals = np.vstack([data_numeric.p3.to_numpy() * np.cos(u1) + data_numeric.p4.to_numpy() * np.sin(u1), data_numeric.p3.to_numpy() * np.cos(u2) + data_numeric.p4.to_numpy() * np.sin(u2)])
ind = np.argmin(P_vals, axis=0)
rows = np.arange(0, u_vals.shape[1], u_vals.shape[1])
u = u_vals[rows, ind]

fig.add_subplot(2, 4, 8)
plt.plot(data_nn.t, data_nn.psi, label='NN', color='red')
plt.plot(data_numeric.t, u_vals[0], label='Numeric', color='black')
plt.plot(data_numeric.t, u_vals[1], label='Numeric', color='black', linestyle='dashdot')

plt.figure()
H_nn = hamiltonian(data_nn,
                   p3=data_nn.p3,
                   p4=data_nn.p4,
                   psi=data_nn.psi)
print(H_nn)
H_analytic_lower_bound = hamiltonian(data_numeric,
                                     p3=data_numeric.p3,
                                     p4=data_numeric.p4,
                                     psi=u_vals[0])
H_analytic_upper_bound = hamiltonian(data_numeric,
                                     p3=data_numeric.p3,
                                     p4=data_numeric.p4,
                                     psi=u_vals[1])
plt.plot(data_nn.t, H_nn, label='NN', color='red')
plt.plot(data_numeric.t, H_analytic_lower_bound, color='black')
plt.plot(data_numeric.t, H_analytic_upper_bound, color='black', linestyle='dashdot')
plt.show()