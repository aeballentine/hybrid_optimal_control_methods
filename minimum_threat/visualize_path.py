import numpy as np
# from shooting_min_threat import cost
from scipy.integrate import solve_ivp
import pandas as pd
from jax import numpy as jnp
from shooting_min_threat import threat


def cost(t_, state):
    x1, x2, p_1, p_2, J = state
    # u = np.arctan2(p_2, p_1)
    u1 = jnp.arctan2(p_2, p_1)
    u2 = u1 + jnp.pi

    # Normalize to [0, 2*pi)
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p_1 * jnp.cos(u1) + p_2 * jnp.sin(u1), p_1 * jnp.cos(u2) + p_2 * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    u = u_vals[ind]

    x1_dot = vel * np.cos(u)
    x2_dot = vel * np.sin(u)

    c_, c_x1, c_x2 = threat(x1, x2, t_)
    p1_dot = -c_x1 / (1 + const_lambda)
    p2_dot = -c_x2 / (1 + const_lambda)

    # psi_dot = (p2_dot * np.cos(u) - p1_dot * np.sin(u)) / (p_1 * np.cos(u) + p_2 * np.sin(u))

    J_dot = (c_ + const_lambda) / (1 + const_lambda)

    return np.array([x1_dot, x2_dot, p1_dot, p2_dot, J_dot])


x1_0 = -15
x2_0 = -15
abserr = 1e-10
relerr = 1e-10
vel = 3.0
const_lambda = 0

sol = np.array([10.444,    0.0206,   0.2079])

x0 = np.array([x1_0, x2_0, sol[1], sol[2], 0])
path_int = solve_ivp(cost, [0, abs(sol[0])],  x0, method='Radau',
                 atol=abserr, rtol=relerr, dense_output=True)

# err = integration(sol)


stoptime = abs(sol[0])
# stoptime = T[0].item()
# x0 = np.array([10, 10, states[0, 2].item(), states[0, 3].item(), states[0, 4].item()])

# int = solve_ivp(system_dynamics, [0, stoptime], x0, method='Radau',
#                 atol=abserr, rtol=relerr, dense_output=True)

t = np.linspace(0, stoptime, 501)
x_state = path_int.sol(t)
x, y, p1, p2, c = x_state

res = pd.DataFrame({'t': t,
                    'x': x,
                    'y': y,
                    'p1': p1,
                    'p2': p2})
res.to_csv(f'ref_path/minimum_q.csv')