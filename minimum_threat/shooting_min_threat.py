import os

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import diffrax
import pandas as pd
import time
from jax import numpy as jnp
import jax
import glob

def threat(x1, x2, t):
    """
    :param x1: x1 coordinate (or list)
    :param x2: x2 coordinate (or list)
    :param t: time
    :param device: device that x1 and x2 are on (cpu, mps, or cuda)
    :return: threat values at (x1, x2) pairs and the gradient in the x1 and x2 direction
    """
    threat_field = 0
    threat_gradient_x1 = 0
    threat_gradient_x2 = 0

    n_peaks = 9
    coeff_peaks = jnp.array([
        # [0.77, 0.6, 0.71, 0.98, 0.87, 0.85, 0.86, 0.81, 0.74],  # intensity
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # intensity
        [-13.86, -3.43, 11.3, -11.71, -2.42, 11.6, -10.76, 4.87, 11.45],  # x location
        [-12.28, -8.35, -10.04, -3.88, 4.29, -13.34, 10.17, 11.58, 8.24],  # y location
        [2.57, 4.92, 4.29, 1.3, 3.7, 3.38, 6.56, 3.21, 2.84],  # x spread
        [3.45, 2.64, 1.76, 2.51, 2.93, 2.72, 2.91, 0.67, 1.26]])  # y spread

    const_1 = 1.0  # 1/2.32

    for m11 in range(n_peaks):
        a = coeff_peaks[0, m11] * (1 + 0.6 * jnp.sin(0.312 * t + 2 * m11) + 0.4 * jnp.cos(1.75 * t + 3 * m11))  # + coeff_peaks[0, m11]

        # drifting centers
        cx = coeff_peaks[1, m11] + 2.0 * (0.2 * jnp.sin(t + m11) + 0.3 * jnp.sin(1.414 * t + 3 * m11) + 0.5 * jnp.cos(1.732 * t + 2 * m11))
        cy = coeff_peaks[2, m11] + 2.0 * (0.6 * jnp.sin(t + m11) + 0.1 * jnp.sin(1.414 * t + 3 * m11) + 0.3 * jnp.cos(1.732 * t + 2 * m11))

        c_xym = jnp.exp(
            - ((x1 - cx) ** 2) / (2 * coeff_peaks[3, m11] ** 2)
            - ((x2 - cy) ** 2) / (2 * coeff_peaks[4, m11] ** 2))
        threat_field = threat_field + a * c_xym

        threat_gradient_x1 = (threat_gradient_x1 -
                              const_1 * ((x1 - cx) / (coeff_peaks[3, m11] ** 2)) *
                              a * c_xym)

        threat_gradient_x2 = (threat_gradient_x2 -
                              const_1 * ((x2 - cy) / (coeff_peaks[4, m11] ** 2)) *
                              a * c_xym)

    threat_field = const_1 * threat_field

    return threat_field, threat_gradient_x1, threat_gradient_x2


def system_dynamics(t_, state, args=None):
    x1, x2, p_1, p_2 = state
    u1 = jnp.arctan2(p_2, p_1)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p_1 * jnp.cos(u1) + p_2 * jnp.sin(u1), p_1 * jnp.cos(u2) + p_2 * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    u = u_vals[ind]

    x1_dot = vel * jnp.cos(u)
    x2_dot = vel * jnp.sin(u)

    _, c_x1, c_x2 = threat(x1, x2, t_)
    p1_dot = -c_x1 / (1 + const_lambda)
    p2_dot = -c_x2 / (1 + const_lambda)

    return jnp.array([x1_dot, x2_dot, p1_dot, p2_dot])


def cost(t_, state):
    x1, x2, p_1, p_2, J = state
    u1 = jnp.arctan2(p_2, p_1)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p_1 * jnp.cos(u1) + p_2 * jnp.sin(u1), p_1 * jnp.cos(u2) + p_2 * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    u = u_vals[ind]

    x1_dot = vel * np.cos(u)
    x2_dot = vel * np.sin(u)

    c_, c_x1, c_x2 = threat(x1, x2, t_)
    p1_dot = -c_x1 / (1 + const_lambda)
    p2_dot = -c_x2 / (1 + const_lambda)

    J_dot = (c_ + const_lambda) / (1 + const_lambda)

    return np.array([x1_dot, x2_dot, p1_dot, p2_dot, J_dot])


def integration(guess):
    tf, p10, p20 = guess
    tf = abs(tf)

    y0 = jnp.array([x1_0, x2_0, p10, p20])
    term = diffrax.ODETerm(system_dynamics)
    solver = diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8))
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=tf,
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10, dtmin=1e-6),
        dt0=1e-5,
        # saveat=diffrax.SaveAt(ts=jnp.linspace(0, tf, 100)),
        max_steps=int(1e5),
        adjoint=diffrax.DirectAdjoint(),
        # throw=False,
    )

    x_, y_, p1_, p2_ = sol.ys[-1]
    u1 = jnp.arctan2(p2_, p1_)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p1_ * jnp.cos(u1) + p2_ * jnp.sin(u1), p1_ * jnp.cos(u2) + p2_ * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    psi_ = u_vals[ind]

    c_, _, _ = threat(x_, y_, tf)
    Hf = (const_lambda + c_) / (1 + const_lambda) + p1_ * vel * np.cos(psi_) + p2_ * vel * np.sin(psi_)
    err = np.array([(x_ - final_state[0]), (y_ - final_state[1]), Hf])

    return err

def solve_bvp(my_args):
    my_type, x_init, path_cost_guess = my_args
    start_time = time.time()

    ans = least_squares(integration, x_init,
                        method='lm',
                        # loss='l1_loss',
                        max_nfev=200,
                        diff_step=1e-6,
                        # jac='3-point'
                        )
    print(' ')
    print(ans)
    runtime = time.time() - start_time

    sol = ans.x
    x0 = np.array([x1_0, x2_0, sol[1], sol[2], 0])
    path_int = solve_ivp(cost, [0, abs(sol[0])],  x0, method='Radau',
                     atol=abserr, rtol=relerr, dense_output=True)

    with open(logfile, 'a') as f:
        if my_type == 'NN':
            f.write("=== Solver Run: NN Initial Guess ===\n")
            f.write(f"Initial cost guess: {path_cost_guess:.4f}\n")
        else:
            f.write("=== Solver Run: Random Initial Guess ===\n")

        f.write(f"Runtime: {runtime:.4f} s\n")
        f.write(f'Number of iterations: {ans.nfev} \n')

        f.write("Initial guess:\t")
        f.write(np.array2string(x_init, precision=4) + "\n")

        f.write("Solution (ans):\t")
        f.write(np.array2string(sol, precision=4) + "\n")

        f.write("Residual (integration(ans)):\t")
        f.write(np.array2string(ans.fun, precision=4) + "\n")

        f.write("Final state:\t")
        f.write(np.array2string(path_int.y[:, -1], precision=4))

        f.write("\n\n ~~~~~~~~~~~~~~~~~~~ \n\n")

    return True


if __name__ == "__main__":
    for _ in range(3):
        for file in glob.glob('trial_three/*.csv'):
            data = pd.read_csv(file)
            my_time = data.t.to_numpy()
            my_ind = np.where(my_time == 0)[0]

            x1_0 = data.x10.to_numpy()[0]
            x2_0 = data.x20.to_numpy()[0]
            psi_0 = data.psi.to_numpy()[my_ind]
            p1_0 = data.p1.to_numpy()[my_ind]
            p2_0 = data.p2.to_numpy()[my_ind]
            magnitude = data.rho.to_numpy()[my_ind]
            vel = data.const_vel.to_numpy()[0]
            const_lambda = data.const_lambda.to_numpy()[0]

            ind_adjusted = np.delete(my_ind,0)
            ind_adjusted = np.append(ind_adjusted, my_ind[0])

            final_time = my_time[ind_adjusted - 1]

            pointwise_cost, _, _ = threat(data.x1.to_numpy(), data.x2.to_numpy(), data.t.to_numpy())
            pointwise_cost = np.asarray(pointwise_cost)
            dt = my_time[1:] - my_time[:-1]
            dt[dt < 0] = 0

            cumulative_cost = np.cumsum(dt * pointwise_cost[:-1])
            cost_by_path = cumulative_cost[ind_adjusted - 1]
            cost_by_path[1:] = cost_by_path[1:] - cost_by_path[:-1]

            final_state = [0.0, 0.0]

            # SET UP FOR THE SOLVER & SOLVE THE SHOOTING PROBLEM
            abserr = 1.0e-10
            relerr = 1.0e-10

            path = 'trial_three_shooting/'
            os.makedirs(path, exist_ok=True)
            logfile = os.path.join(path, f'{x1_0}_{x2_0}_analytic_dynamic_random.txt')

            if logfile in glob.glob(path + '*'):
                print('skipping')
                continue

            t_guess = (x1_0**2 + x2_0**2) ** 0.5 / vel

            if __name__ == '__main__':
                num_trials = 10
                jax.config.update("jax_enable_x64", True)

                args = [("NN", np.array([val, p1_0[j], p2_0[j]]), cost_by_path[j]) for j, val in enumerate(final_time)] + [('random', np.random.uniform([t_guess, -1, -1], [2.0 * t_guess, 1, 1]), 0.0) for _ in range(num_trials)]
                for entry in args:
                    solve_bvp(entry)

                print('Done.')