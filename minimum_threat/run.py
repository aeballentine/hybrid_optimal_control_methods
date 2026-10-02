import torch
import argparse
import time
from torch import nn
import numpy as np

from utils_threat import NeuralNetwork, save_res, threat
from training_min_threat_multi import losses


def train(
        args,
        initial_condition
):
    velocity = args.velocity
    final_state = torch.tensor(args.final_condition, device=device)
    num_colloc = args.num_colloc
    num_paths = args.num_paths
    lambda_coeff = args.lamb
    start_state = torch.tensor(initial_condition, device=device, dtype=torch.float32)
    x_training = start_state[0].view(-1).repeat_interleave(num_colloc).view(-1, 1)
    y_training = start_state[1].view(-1).repeat_interleave(num_colloc).view(-1, 1)

    tf_guess = torch.linspace(1.25, 2, 4, dtype=torch.float32, device=device) * (start_state ** 2).sum() ** 0.5 / velocity
    tfNet = nn.Parameter(tf_guess, requires_grad=True)

    state_dims = [1, 400, 400, 16]
    state_activation = nn.Tanh()
    StateNet = NeuralNetwork(state_dims, function=state_activation).to(device)

    params = list(StateNet.parameters())
    opt = torch.optim.Adam(params, lr=args.learning_rate)

    if args.type == 'dynamic_unbounded':
        print(f'\t \t\t Total Loss \t  H \t\t\t H_u \t\t\t p1 \t\t\t p2 \t\t\t x1 \t\t\t x2 \t\t\t x(i) \t\t\t x(f) \t\t\t Cost \t\t\t H')
    else:
        raise Exception('Not implemented yet')

    lr_decay = args.lr_decay
    alpha = 0.8

    training_times = torch.linspace(0, 1, num_colloc).repeat(num_paths).view(-1, 1).to(device)
    costate_multi = 1
    for epoch in range(args.epochs_adam + 1):
        training_times.requires_grad = True
        x_training.requires_grad = True
        y_training.requires_grad = True

        opt.zero_grad()
        curr_loss, other_losses, tf = losses(model_states=StateNet, model_tf=tfNet, x=x_training, y=y_training, t=training_times, device=device, velocity=velocity, final_state=final_state, lamb=lambda_coeff, costate_multiplier=costate_multi, num_colloc=num_colloc)

        if epoch != 0:
            ratio = torch.exp(curr_loss / loss_prior).clip(-500, 500) + 0.5 * torch.exp(curr_loss / 1e-4).clip(-500, 500)
            lambdi = ratio / torch.sum(ratio)
            lambdi = lambdi.detach()
            lambdas = alpha * lambdas + (1 - alpha) * lambdi.detach()
        else:
            loss_prior = torch.ones_like(curr_loss, device=device)
            lambdas = torch.ones_like(curr_loss, device=device)
            continue
        net_loss = torch.sum(lambdas * curr_loss) + lambdas.min() * torch.sum(other_losses)
        net_loss.backward()
        opt.step()

        if epoch % 5000 == 0 and epoch < 12000:
            for param_group in opt.param_groups:
                param_group['lr'] *= lr_decay
        if epoch == 500:
            opt.add_param_group({'params': [tfNet], 'lr': 1e-3})
            costate_multi = 1
        if epoch == 5000:
            costate_multi = 10

        if epoch % 500 == 0:
            print(f'Epoch = {epoch:4.0f}: {net_loss.item():.2e}\t\t ',
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in curr_loss),
                  '\n',
                  *(f"{my_loss.item():.2e}\t\t" for my_loss in other_losses),
                  '\n',
                  *(f"{final_time.item():.2e}\t\t" for final_time in tf),
                  '\n')

    return StateNet, tfNet


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='dynamic_unbounded')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--bounds', type=float, nargs=4, default=[-15.0, -15.0, 15.0, 15.0])    # x l.b., y l.b., x u.b., y u.b.
    parser.add_argument('--num_steps', type=int, default=7)
    parser.add_argument('--final_condition', type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument('--velocity', type=float, default=3.0)
    parser.add_argument('--epochs_adam', type=int, default=10000)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_paths', type=float, default=1)
    parser.add_argument('--num_colloc', type=int, default=512)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--file_prefix', type=str, default='trial_three')

    my_args = parser.parse_args()
    x_arr = np.linspace(my_args.bounds[0], my_args.bounds[2], my_args.num_steps).astype(np.float32)
    y_arr = np.linspace(my_args.bounds[1], my_args.bounds[3], my_args.num_steps).astype(np.float32)
    X, Y = np.meshgrid(x_arr, y_arr)
    X, Y = X.flatten(), Y.flatten()
    for i, x_init in enumerate(X):
        ic = [x_init, Y[i]]
        x, y = ic[0], ic[1]
        lamb = my_args.lamb
        file_prefix = 'trial_three'
        my_vel = my_args.velocity

        start_time = time.time()
        state_model, tf_model = train(my_args, ic)
        end_time = time.time()
        print("Training done in {:.1f}s".format(end_time - start_time))

        path = rf'{my_args.file_prefix}/{x:.1f}_{y:.1f}_NN_path.csv'
        save_res(x0=x, y0=y,
                 model_states=state_model,
                 model_tf=tf_model,
                 dt=0.05,
                 save_path=path,
                 device=device,
                 threat_func=threat,
                 lamb=lamb,
                 training_time=end_time - start_time,
                 vel=my_vel)