import torch

from utils import threat
from torch.func import jacrev

loss_function = torch.nn.HuberLoss(delta=0.1)
def losses(model_states, model_tf, x, y, t, device, velocity, final_state, lamb, costate_multiplier, num_colloc):
    def model(t_i):
        return model_states(t_i.unsqueeze(0)).squeeze(0)

    # get the model prediction and extract all the relevant variables (x1, x2, p1, p2, psi - heading angle, and predicted time to reach the goal
    final_time_path = model_tf
    states = model_states(t)
    grads = torch.vmap(jacrev(model))(t)

    x1 = states[:, :, 0]
    x2 = states[:, :, 1]
    psi = states[:, :, 2]
    magnitude = states[:, :, 3]
    p1 = magnitude * torch.cos(psi)
    p2 = magnitude * torch.sin(psi)

    x1_dot = grads[:, :, 0, 0]
    x2_dot = grads[:, :, 1, 0]
    psi_dot = grads[:, :, 2, 0]
    magnitude_dot = grads[:, :, 3, 0]

    # manually calculate p1 and p2
    p1_dot = magnitude_dot * torch.cos(psi) - magnitude * torch.sin(psi) * psi_dot
    p2_dot = magnitude_dot * torch.sin(psi) + magnitude * torch.cos(psi) * psi_dot

    t_full = t.repeat((1, 4))

    # get the threat field prediction at each state
    c, c_x1, c_x2 = threat(x1, x2, t=final_time_path * t_full, device=device)
    hamiltonian = magnitude * velocity + c + lamb
    h_min = torch.mean(hamiltonian)

    # dH/du = 0
    control_eq = -p1 * velocity * torch.sin(psi) + p2 * velocity * torch.cos(psi)
    h_u_loss = torch.mean(control_eq ** 2)

    # derivative losses
    p1_dot_exact = -final_time_path * c_x1 / (lamb + 1)
    p2_dot_exact = -final_time_path * c_x2 / (lamb + 1)
    p1_loss = loss_function(p1_dot, p1_dot_exact)
    p2_loss = loss_function(p2_dot, p2_dot_exact)

    x1_dot_exact = final_time_path * velocity * torch.cos(psi)
    x2_dot_exact = final_time_path * velocity * torch.sin(psi)
    x1_loss = loss_function(x1_dot, x1_dot_exact)
    x2_loss = loss_function(x2_dot, x2_dot_exact)

    # cost prediction along the path
    dt = t[1] - t[0]
    net_cost = final_time_path * c.sum(dim=0) * dt

    # enforce the final state prediction
    predicted_final_state = torch.cat([x1[-1, :].view(-1, 1), x2[-1, :].view(-1, 1)], dim=1)
    final_loss = loss_function(predicted_final_state, final_state.repeat(4, 1))

    # enforce that the initial state should be equal
    predicted_initial_state = torch.cat([x1[0, :].view(-1, 1), x2[0, :].view(-1, 1)], dim=1)
    initial_loss = loss_function(predicted_initial_state, torch.cat([x, y], dim=1)[:4])

    # Hamiltonian final loss
    cf, _, _ = threat(final_state[0], final_state[1], final_time_path, device)
    rho_f = magnitude[-1, :]
    h_target = torch.zeros((4, ), device=final_state.device)
    hamiltonian_loss = loss_function(rho_f + cf / velocity, h_target)

    return (torch.cat([hamiltonian_loss.view(-1, 1),
                       h_u_loss.view(-1, 1),
                       p1_loss.view(-1, 1),
                       p2_loss.view(-1, 1),
                       x1_loss.view(-1, 1),
                       x2_loss.view(-1, 1),
                       initial_loss.view(-1, 1),
                       final_loss.view(-1, 1),
                       # net_cost.view(-1, 1)
                       ]),
            torch.cat([
                net_cost.view(-1, 1),
                # -final_time_path.view(-1, 1)
                h_min.view(-1, 1)
                       ]),
            final_time_path
            )