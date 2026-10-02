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


def system_dynamics(t_, state, args=None):
    x1, x2, x3, x4, p_2, p_3, p_4 = state
    # jax.debug.print('state: {val}', val=state)
    u1 = jnp.arctan2(p_4, p_3)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p_3 * jnp.cos(u1) + p_4 * jnp.sin(u1), p_3 * jnp.cos(u2) + p_4 * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    u = u_vals[ind]

    m = m0 - mdot * t_
    D = 0.5 * rho * jnp.exp(-x2 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    cos_gamma = x3 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5
    sin_gamma = x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5
    x1_dot = x3
    x2_dot = x4
    x3_dot = T * jnp.cos(u) / m - D * cos_gamma / m
    x4_dot = T * jnp.sin(u) / m - D * sin_gamma / m - g

    Dx2 = -0.5 * rho * jnp.exp(-x2 / h0) * (1 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    Dx3 = rho * jnp.exp(-x2 / h0) * x3 * S * Cd
    Dx4 = rho * jnp.exp(-x2 / h0) * x4 * S * Cd
    sgamma_x3 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    cgamma_x3 = x4 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)

    sgamma_x4 = x3 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    cgamma_x4 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)

    p2_dot = p_3 * cos_gamma / m * Dx2 + p_4 * sin_gamma / m * Dx2
    p3_dot = p_3 / m * (Dx3 * cos_gamma + D * cgamma_x3) + p_4 / m * (Dx3 * sin_gamma + D * sgamma_x3)
    p4_dot = -p_2 + p_3 / m * (Dx4 * cos_gamma + D * cgamma_x4) + p_4 / m * (Dx4 * sin_gamma + D * sgamma_x4)

    # jax.debug.print('{val}', val=jnp.array([D, m]))
    # jax.debug.print('derivative: {arr}', arr=jnp.array([x1_dot, x2_dot, x3_dot, x4_dot, p2_dot, p3_dot, p4_dot]))

    return jnp.array([x1_dot, x2_dot, x3_dot, x4_dot, p2_dot, p3_dot, p4_dot])


def cost(t_, state):
    x1, x2, x3, x4, p_2, p_3, p_4, J = state
    u1 = jnp.arctan2(p_4, p_3)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p_3 * jnp.cos(u1) + p_4 * jnp.sin(u1), p_3 * jnp.cos(u2) + p_4 * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    u = u_vals[ind]

    m = m0 - mdot * t_
    D = 0.5 * rho * jnp.exp(-x2 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    cos_gamma = x3 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5
    sin_gamma = x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5
    x1_dot = x3
    x2_dot = x4
    x3_dot = T * jnp.cos(u) / m - D * cos_gamma / m
    x4_dot = T * jnp.sin(u) / m - D * sin_gamma / m - g

    Dx2 = -0.5 * rho * jnp.exp(-x2 / h0) * (1 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    Dx3 = rho * jnp.exp(-x2 / h0) * x3 * S * Cd
    Dx4 = rho * jnp.exp(-x2 / h0) * x4 * S * Cd
    sgamma_x3 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    cgamma_x3 = x4 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)

    sgamma_x4 = x3 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    cgamma_x4 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)

    p2_dot = p_3 * cos_gamma / m * Dx2 + p_4 * sin_gamma / m * Dx2
    p3_dot = p_3 / m * (Dx3 * cos_gamma + D * cgamma_x3) + p_4 / m * (Dx3 * sin_gamma + D * sgamma_x3)
    p4_dot = -p_2 + p_3 / m * (Dx4 * cos_gamma + D * cgamma_x4) + p_4 / m * (Dx4 * sin_gamma + D * sgamma_x4)

    J_dot = 1

    return np.array([x1_dot, x2_dot, x3_dot, x4_dot, p2_dot, p3_dot, p4_dot, J_dot])


def integration(guess):
    tf, p20, p30, p40 = guess
    tf = abs(tf)
    tf = min(tf, 2)
    # print(guess)
    # print(' ')

    y0 = jnp.array([x1_0, x2_0, x3_0, x4_0, p20, p30, p40])
    term = diffrax.ODETerm(system_dynamics)
    solver = diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-10, atol=1e-10))
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=tf,
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10, dtmin=1e-6),    # can include dtmin
        dt0=1e-6,
        # saveat=diffrax.SaveAt(ts=jnp.linspace(0, tf, 100)),
        max_steps=int(1e5),
        adjoint=diffrax.DirectAdjoint(),
        throw=False,
    )

    x1_, x2_, x3_, x4_, p2_, p3_, p4_ = sol.ys[-1]
    u1 = jnp.arctan2(p4_, p3_)
    u2 = u1 + jnp.pi
    u_vals = jnp.array([u1 % (2 * jnp.pi), u2 % (2 * jnp.pi)])
    P_vals = jnp.array([p3_ * jnp.cos(u1) + p4_ * jnp.sin(u1), p3_ * jnp.cos(u2) + p4_ * jnp.sin(u2)])
    ind = jnp.argmin(P_vals)
    psi_ = u_vals[ind]

    D = 0.5 * rho * np.exp(-x2_ / h0) * (x3_ ** 2 + x4_ ** 2) * S * Cd
    cos_gamma = x3_ / (x3_ ** 2 + x4_ ** 2) ** 0.5
    sin_gamma = x4_ / (x3_ ** 2 + x4_ ** 2) ** 0.5
    m = m0 - mdot * tf
    Hf = 1 + p2_ * x4_ + p3_ * (T * np.cos(psi_) - D * cos_gamma) / m + p4_ * ((T * np.sin(psi_) - D * sin_gamma) / m - g)
    err = np.array([(x2_ - final_state[0]), (x3_ - final_state[1]), (x4_ - final_state[2]), Hf])

    return err

def solve_bvp(my_args):
    my_type, x_init, path_cost_guess = my_args
    start_time = time.time()

    ans = least_squares(integration, x_init,
                        method='lm',
                        # loss='l1_loss',
                        max_nfev=300,
                        diff_step=1e-5,
                        # jac='3-point'
                        )
    print(' ')
    print(ans)
    runtime = time.time() - start_time

    sol = ans.x
    x0 = np.array([x1_0, x2_0, x3_0, x4_0, sol[1], sol[2], sol[3], 0])
    path_int = solve_ivp(cost, [0, abs(sol[0])],  x0, method='Radau',
                     atol=abserr, rtol=relerr, dense_output=True)

    with open(logfile, 'a') as f:
        if my_type == 'NN':
            f.write("=== Solver Run: NN Initial Guess ===\n")
            # f.write(f"Initial cost guess: {path_cost_guess:.4f}\n")
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

    # if all(ans.fun < 1e-5):
    if my_type == 'NN':
        stoptime = abs(sol[0])

        t = np.linspace(0, stoptime, 501)
        x_state = path_int.sol(t)
        x1, x2, x3, x4, p2, p3, p4, c = x_state

        res = pd.DataFrame({'t': t,
                            'x1': x1,
                            'x2': x2,
                            'x3': x3,
                            'x4': x4,
                            'p2': p2,
                            'p3': p3,
                            'p4': p4})
        res.to_csv(f'{path}rocket_launch_hybrid_solution.csv')
    return True


if __name__ == "__main__":
    # SET UP FOR THE SOLVER & SOLVE THE SHOOTING PROBLEM
    x1_0, x2_0, x3_0, x4_0 = 0, 0, 0, 0
    final_state = np.array([1, 6.278933, 0])
    T = 2.093217
    S = 218.1790e-12
    rho = 61.05110e9
    g = 1.144258
    m0 = 1
    mdot = 0.6
    Cd = 0.5
    h0 = 46.88889e-3
    # T = 2.1e6
    # S = 7.069
    # rho = 1.225
    # g = 9.81
    # final_state = np.array([180, 7.8e3, 0])
    # m0 = 1.1702e5
    # mdot = 807.6
    # Cd = 0.5
    # h0 = 8.44

    abserr = 1.0e-10
    relerr = 1.0e-10

    path = 'ref_path_transfer/'
    os.makedirs(path, exist_ok=True)
    logfile = os.path.join(path, f'rocket_launch_final.txt')

    # x_guess_nn = np.array([0.978, -0.8129, -0.0297, -0.5766])
    # x_guess_nn = np.array([0.99, -0.051, -0.0015, -0.0249])
    nn_solution = pd.read_csv('ref_path_transfer/rocket_launch_nn_solution_3.csv')
    x_guess_nn = np.array([nn_solution.t.to_numpy()[-1], nn_solution.p2.to_numpy()[0], nn_solution.p3.to_numpy()[0], nn_solution.p4.to_numpy()[0]])
    # x_guess_nn = np.array([1.4810895, -0.11892822, -0.01640133, -0.07383248])

    # if logfile in glob.glob(path + '*'):
    #     print('skipping')
    #     continue

    t_min = 0
    t_max = 2

    if __name__ == '__main__':
        num_trials = 10
        jax.config.update("jax_enable_x64", True)

        # args = [("NN", np.array([val, p1_0[j], p2_0[j]]), cost_by_path[j]) for j, val in enumerate(final_time)] +
        args = [('NN', x_guess_nn, None)] + [('random', np.random.uniform([t_min, -1, -1, -1], [t_max, 1, 1, 1]), 0.0) for _ in range(num_trials)]
        for entry in args:
            solve_bvp(entry)

        print('Done.')