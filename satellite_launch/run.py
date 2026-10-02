import torch
import argparse
import time
from torch import nn
import numpy as np
import pandas as pd

from utils import NeuralNetworkSimple
from training_rocket_launch import losses


def train(
        args,
        initial_condition
):
    num_colloc = args.num_colloc
    start_state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    final_state = torch.tensor([1, 6.278933, 0], device=device, dtype=torch.float32)
    # final_state = torch.tensor([180, 7.8e3, 0], device=device, dtype=torch.float32)

    tf_dims = [4, 64, 1]
    tf_activation = nn.Tanh()
    tfNet = NeuralNetworkSimple(tf_dims, function=tf_activation, output_activation=nn.Sigmoid()).to(device)

    # tf_guess = torch.tensor([0.4]).to(device)
    # tfNet = nn.Parameter(tf_guess, requires_grad=True)
    # state_dims = [1, 400, 500, 500, 400, 7]
    state_dims = [1, 150, 300, 150, 7]
    state_activation = nn.Tanh()    # this is Sigmoid() or Tanh()
    StateNet = NeuralNetworkSimple(state_dims, function=state_activation).to(device)
    # StateNet = NeuralEnsemble().to(device)
    # StateNet = IndependentBranchNet(n_inputs=1, m_groups=4, i_components=4, h1=100, h2=100).to(device)

    params = list(StateNet.parameters()) + list(tfNet.parameters())
    opt = torch.optim.Adam(params, lr=args.learning_rate)
    # opt.add_param_group({'params': [tfNet], 'lr': 1e-2})

    # if args.type == 'static unbounded':
        # losses = static_unbounded_loss
        # print(f'\t \t\t Total Loss \t  H \t\t\t H_u \t\t\t p1 \t\t\t p2 \t\t\t x1 \t\t\t x2 \t\t\t x(i) \t\t\t x(f) \t\t\t Cost')
    if args.type == 'dynamic_unbounded':
        print(f'\t \t\t Total Loss \t  H \t\t\t H_u \t\t\t p2 \t\t\t p3 \t\t\t p4 \t\t\t x1 \t\t\t x2 \t\t\t x3 \t\t\t x4 \t\t\t x(i) \t\t\t x(f) \t\t\t Cost')
        # losses = dynamic_unbounded_loss
    # elif args.type == 'static bounded':
    #     losses = static_bounded_loss
    #     print(f'\t \t\t Total Loss \t  H \t\t\t H_u \t\t\t p1 \t\t\t p2 \t\t\t p3 \t\t\t x1 \t\t\t x2 \t\t\t th \t\t\t x(i) \t\t\t x(f) \t\t\t p3(f) \t\t\t Cost')
    else:
        raise Exception('Not implemented yet')

    lr_decay = args.lr_decay
    # n_epochs = args.epochs
    alpha = 0.9

    training_times = torch.linspace(0, 1, num_colloc).to(device).view(-1, 1)
    # training_times = training_times.repeat((4, 1))
    # del args
    # u_min = torch.linspace(0, 2 * torch.pi, num_paths + 1, dtype=torch.float32)[:-1].repeat_interleave(num_colloc).view(-1, 1).to(device)
    # u_max = torch.linspace(0, 2 * torch.pi, num_paths + 1, dtype=torch.float32)[1:].repeat_interleave(num_colloc).view(-1, 1).to(device)
    # initial_state = torch.tensor(initial_condition, device=device)
    costate_multi = 1
    for epoch in range(args.epochs_adam + 1):
        # if start_state is None:
        #     x_training = (-15 + 30 * torch.rand(num_paths, device=device))
        #     x_training = x_training.repeat_interleave(num_colloc).view(-1, 1)
        #     y_training = (-15 + 30 * torch.rand(num_paths, device=device))
        #     y_training = y_training.repeat_interleave(num_colloc).view(-1, 1)

        training_times.requires_grad = True
        # x_training.requires_grad = True
        # y_training.requires_grad = True
        # u_min.requires_grad = True
        # u_max.requires_grad = True

        opt.zero_grad()
        curr_loss, other_losses = losses(model_states=StateNet, model_tf=tfNet, t=training_times, final_state=final_state, costate_multiplier=1, initial_state=start_state)
        # curr_loss, other_losses, tf = losses(model_states=StateNet, model_tf=tfNet, u_min=u_min, u_max=u_max, t=training_times, device=device, velocity=velocity, final_state=final_state, lamb=lambda_coeff, num_paths=num_paths, initial_state=initial_state)

        if epoch != 0:
            ratio = torch.exp(curr_loss / loss_prior).clip(-500, 500) + 0.5 * torch.exp(curr_loss / 1e-4).clip(-500, 500)
            # ratio = 0.5 * torch.exp(curr_loss / 1e-4).clip(-500, 500)
            lambdi = ratio / torch.sum(ratio)
            lambdi = lambdi.detach()
            lambdas = alpha * lambdas + (1 - alpha) * lambdi.detach()
        else:
            loss_prior = torch.ones_like(curr_loss, device=device)
            lambdas = torch.ones_like(curr_loss, device=device)
            continue
        net_loss = torch.sum(lambdas * curr_loss) # + lambdas.min() * torch.sum(other_losses)
        # if torch.sum(curr_loss.detach()) < 1e-1:
        # net_loss = net_loss + lambdas.mean() * torch.sum(other_losses)
        net_loss.backward()
        opt.step()

        if epoch % 2500 == 0 and epoch < 12000:
            for param_group in opt.param_groups:
                param_group['lr'] *= lr_decay
        # if epoch == 500:
        #     opt.add_param_group({'params': [tfNet], 'lr': 1e-3})
        #     costate_multi = 1
        if epoch == 5000:
            costate_multi = 10

        if epoch % 500 == 0:
            print(f'Epoch = {epoch:4.0f}: {net_loss.item():.2e}\t\t ',
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in curr_loss),
                  '\n',
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in other_losses),
                  '\n',
                  # *(f"{final_time.item():.2e}\t\t" for final_time in tf),
                  # '\n'
                  )

    def closure():
        nonlocal use_other, weights
        opt.zero_grad()
        training_times.requires_grad = True
        # x_training.requires_grad = True
        # y_training.requires_grad = True

        c_loss, o_loss = losses(model_states=StateNet, model_tf=tfNet, t=training_times, final_state=final_state, costate_multiplier=1, initial_state=start_state)
        if use_other:
            n_loss = c_loss.sum() + o_loss.sum()
        else:
            n_loss = (c_loss * weights).sum()

        n_loss.backward()
        return n_loss
    #
    params = list(StateNet.parameters()) + list(tfNet.parameters())
    opt = torch.optim.LBFGS(params,
                            lr=1.0,
                            history_size=50,
                            max_iter=15,
                            line_search_fn="strong_wolfe"
                            )
    weights = torch.tensor([10, 1, 100, 100, 1, 1, 1, 1, 10, 1], dtype=torch.float32).to(device)
    use_other = False
    for epoch in range(args.epochs_lbfgs):
        opt.step(closure)
        curr_loss, other_losses = losses(model_states=StateNet, model_tf=tfNet, t=training_times, final_state=final_state, costate_multiplier=1, initial_state=start_state)
        if epoch % 50 == 0:
            print(f'Epoch = {epoch:4.0f}: {curr_loss.sum().item():.2e}\t\t ',
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in curr_loss),
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in other_losses), )
    #     if curr_loss.sum() < 5e-5:
    #         break

    # use_other = True
    # for epoch in range(args.epochs_lbfgs_2):
    #     opt.step(closure)
    #     curr_loss, other_losses = losses(model_states=StateNet, model_tf=tfNet, x=x_training, y=y_training, t=training_times, device=device, velocity=velocity, final_state=final_state, lamb=lambda_coeff)
    #     if epoch % 50 == 0:
    #         print(f'Epoch = {epoch:4.0f}: {curr_loss.sum().item():.2e}\t\t ',
    #               *(f"{my_loss.item():.2e}\t\t" for my_loss in curr_loss),
    #               *(f"{my_loss.item():.2e}\t\t" for my_loss in other_losses), )
    final_time = 2 * tfNet(start_state)
    result = StateNet(training_times)

    T = 2.093217
    S = 218.1790e-12
    rho = 61.05110e9
    g = 1.144258
    m0 = 1
    mdot = 0.6

    Cd = 0.5
    h0 = 46.88889e-3

    x1 = result[:, 0].view(-1, 1)
    x2 = result[:, 1].view(-1, 1)
    x3 = result[:, 2].view(-1, 1)
    x4 = result[:, 3].view(-1, 1)

    psi = result[:, 4].view(-1, 1)
    p2 = result[:, 5].view(-1, 1)
    magnitude = result[:, 6].view(-1, 1)
    #
    p3 = magnitude * torch.cos(psi)
    p4 = magnitude * torch.sin(psi)
    # p3 = result[:, 6].view(-1, 1)
    # p4 = result[:, 7].view(-1, 1)

    m = m0 - mdot * final_time * training_times
    D = 0.5 * rho * torch.exp(-x2 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    cos_gamma = x3 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5
    sin_gamma = x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** 0.5

    # Dx2 = -0.5 * rho * torch.exp(-x2 / h0) * (1 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    # Dx3 = rho * torch.exp(-x2 / h0) * x3 * S * Cd
    # Dx4 = rho * torch.exp(-x2 / h0) * x4 * S * Cd
    # sgamma_x3 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    # cgamma_x3 = x4 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    #
    # sgamma_x4 = x3 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)
    # cgamma_x4 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-6) ** (3 / 2)

    hamiltonian = 1 + p2 * x4 + p3 * (T * torch.cos(psi) - D * cos_gamma) / m + p4 * ((T * torch.sin(psi) - D * sin_gamma) / m - g)
    print(hamiltonian[-1])
    print(result[-1, :])
    # print(result)
    # print([result[6] * torch.cos(result[4]), result[6] * torch.sin(result[4])])
    res_dataframe = pd.DataFrame({
        't': (final_time * training_times).view(-1).detach().cpu().numpy(),
        'x1': result[:, 0].view(-1).detach().cpu().numpy(),
        'x2': result[:, 1].view(-1).detach().cpu().numpy(),
        'x3': result[:, 2].view(-1).detach().cpu().numpy(),
        'x4': result[:, 3].view(-1).detach().cpu().numpy(),
        'psi': result[:, 4].view(-1).detach().cpu().numpy(),
        'p2': result[:, 5].view(-1).detach().cpu().numpy(),
        'magnitude': result[:, 6].detach().cpu().numpy(),
        'p3': p3.view(-1).detach().cpu().numpy(),
        'p4': p4.view(-1).detach().cpu().numpy()
    })
    res_dataframe.to_csv('ref_path_transfer/rocket_launch_nn_solution_3.csv')
    return StateNet, tfNet


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='dynamic_unbounded')  # 1e-4 baseline to train initial model
    parser.add_argument('--learning_rate', type=float, default=3e-4)  # 1e-4 baseline to train initial model
    parser.add_argument('--bounds', type=float, nargs=4, default=[-15.0, -15.0, 15.0, 15.0])    # x l.b., y l.b., x u.b., y u.b.
    parser.add_argument('--num_steps', type=int, default=7)
    parser.add_argument('--final_condition', type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument('--velocity', type=float, default=3.0)
    parser.add_argument('--epochs_adam', type=int, default=7500)    # was 10,000
    parser.add_argument('--epochs_lbfgs', type=int, default=0)
    # parser.add_argument('--epochs_lbfgs_2', type=int, default=500)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--num_paths', type=float, default=0.9)
    parser.add_argument('--num_colloc', type=int, default=512)  # was 512 originally
    parser.add_argument('--lamb', type=float, default=0.0)

    my_args = parser.parse_args()
    train(my_args, None)

    # x_arr = np.linspace(my_args.bounds[0], my_args.bounds[2], my_args.num_steps).astype(np.float32)
    # y_arr = np.linspace(my_args.bounds[1], my_args.bounds[3], my_args.num_steps).astype(np.float32)
    # X, Y = np.meshgrid(x_arr, y_arr)
    # X, Y = X.flatten(), Y.flatten()
    # # X, Y = [-15.0], [-15.0]
    # for i, x_init in enumerate(X):
    #     ic = [x_init, Y[i]]
    #     x, y = ic[0], ic[1]
    #     lamb = my_args.lamb
    #     # file_prefix = my_args.type
    #     file_prefix = 'trial_two'
    #     my_vel = my_args.velocity
    #
    #     start_time = time.time()
    #     state_model, tf_model = train(my_args, ic)
    #     end_time = time.time()
    #     print("Training done in {:.1f}s".format(end_time - start_time))
    #
    #     path = rf'{file_prefix}/{x:.1f}_{y:.1f}_NN_path.csv'
    #     save_res(x0=x, y0=y,
    #              model_states=state_model,
    #              model_tf=tf_model,
    #              dt=0.5,
    #              save_path=path,
    #              device=device,
    #              threat_func=threat,
    #              lamb=lamb,
    #              training_time=end_time - start_time,
    #              vel=my_vel)