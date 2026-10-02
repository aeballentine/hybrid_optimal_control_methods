import torch

T = 2.093217
S = 218.1790e-12
rho = 61.05110e9
g = 1.144258
m0 = 1
mdot = 0.6
Cd = 0.5
h0 = 46.88889e-3
loss_function = torch.nn.HuberLoss(delta=0.1)
# T = 2.1e6
# S = 7.069
# rho = 1.225
# g = 9.81
# m0 = 1.1702e5
# mdot = 807.6
# Cd = 0.5
# h0 = 8.44

def losses(model_states, model_tf, t, final_state, initial_state, costate_multiplier=1):
    # get the model prediction and extract all the relevant variables (x1, x2, p1, p2, psi - heading angle, and predicted time to reach the goal

    # tf = torch.sigmoid(model_tf)
    tf = 2 * model_tf(initial_state).view(-1)
    # tf = model_tf
    states = model_states(t)

    x1 = states[:, 0].view(-1, 1)
    x2 = states[:, 1].view(-1, 1)
    x3 = states[:, 2].view(-1, 1)
    x4 = states[:, 3].view(-1, 1)

    psi = states[:, 4].view(-1, 1)
    p2 = states[:, 5].view(-1, 1)
    magnitude = states[:, 6].view(-1, 1)
    #
    p3 = magnitude * torch.cos(psi)
    p4 = magnitude * torch.sin(psi)
    # p3 = states[:, 6].view(-1, 1)
    # p4 = states[:, 7].view(-1, 1)

    m = m0 - mdot * tf * t
    D = 0.5 * rho * torch.exp(-x2 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    cos_gamma = x3 / (x3 ** 2 + x4 ** 2 + 1e-10) ** 0.5
    sin_gamma = x4 / (x3 ** 2 + x4 ** 2 + 1e-10) ** 0.5

    Dx2 = -0.5 * rho * torch.exp(-x2 / h0) * (1 / h0) * (x3 ** 2 + x4 ** 2) * S * Cd
    Dx3 = rho * torch.exp(-x2 / h0) * x3 * S * Cd
    Dx4 = rho * torch.exp(-x2 / h0) * x4 * S * Cd
    sgamma_x3 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-10) ** (3 / 2)
    cgamma_x3 = x4 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-10) ** (3 / 2)

    sgamma_x4 = x3 ** 2 / (x3 ** 2 + x4 ** 2 + 1e-10) ** (3 / 2)
    cgamma_x4 = -x3 * x4 / (x3 ** 2 + x4 ** 2 + 1e-10) ** (3 / 2)

    hamiltonian = 1 + p2 * x4 + p3 * (T * torch.cos(psi) - D * cos_gamma) / m + p4 * ((T * torch.sin(psi) - D * sin_gamma) / m - g)
    h_target = torch.tensor([0.0], device=final_state.device)
    h_loss = loss_function(hamiltonian[-1], h_target)   # this was 10

    # dH/du = 0 => hard enforced
    dh_du = p3 * torch.sin(psi) - p4 * torch.cos(psi)
    dh_du_target = torch.zeros_like(dh_du, device=final_state.device)
    hu_loss = loss_function(dh_du, dh_du_target)

    # constraints on the states and co-states
    p2_dot = torch.autograd.grad(p2, t, grad_outputs=torch.ones_like(p2), create_graph=True, retain_graph=True)[0]
    p3_dot = torch.autograd.grad(p3, t, grad_outputs=torch.ones_like(p3), create_graph=True, retain_graph=True)[0]
    p4_dot = torch.autograd.grad(p4, t, grad_outputs=torch.ones_like(p4), create_graph=True, retain_graph=True)[0]

    p2_dot_exact = p3 * cos_gamma / m * Dx2 + p4 * sin_gamma / m * Dx2
    p3_dot_exact = p3 / m * (Dx3 * cos_gamma + D * cgamma_x3) + p4 / m * (Dx3 * sin_gamma + D * sgamma_x3)
    p4_dot_exact = -p2 + p3 / m * (Dx4 * cos_gamma + D * cgamma_x4) + p4 / m * (Dx4 * sin_gamma + D * sgamma_x4)

    p2_loss = loss_function(p2_dot, tf * p2_dot_exact)
    p3_loss = loss_function(p3_dot, tf * p3_dot_exact)
    p4_loss = loss_function(p4_dot, tf * p4_dot_exact)

    x1_dot = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(x1), create_graph=True, retain_graph=True)[0]
    x2_dot = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(x2), create_graph=True, retain_graph=True)[0]
    x3_dot = torch.autograd.grad(x3, t, grad_outputs=torch.ones_like(x3), create_graph=True, retain_graph=True)[0]
    x4_dot = torch.autograd.grad(x4, t, grad_outputs=torch.ones_like(x4), create_graph=True, retain_graph=True)[0]

    x1_dot_exact = x3
    x2_dot_exact = x4
    x3_dot_exact = T * torch.cos(psi) / m - D * cos_gamma / m
    x4_dot_exact = T * torch.sin(psi) / m - D * sin_gamma / m - g

    x1_loss = 10 * loss_function(x1_dot, tf * x1_dot_exact)  # 10 for all loss functions
    x2_loss = 10 * loss_function(x2_dot, tf * x2_dot_exact)
    x3_loss = 10 * loss_function(x3_dot, tf * x3_dot_exact)
    x4_loss = 10 * loss_function(x4_dot, tf * x4_dot_exact)

    # cost prediction along the path
    net_cost = tf

    # enforce the final state prediction
    predicted_final_state = states[-1, [1, 2, 3]]
    final_loss = 20 * loss_function(predicted_final_state, final_state) # 20

    # enforce that the initial state should be equal
    predicted_initial_state = states[0, [0, 1, 2, 3]]
    initial_loss = 20 * loss_function(predicted_initial_state, initial_state)   # 30

    return (torch.cat([h_loss.view(-1, 1),
                       hu_loss.view(-1, 1),
                       p2_loss.view(-1, 1),
                       p3_loss.view(-1, 1),
                       p4_loss.view(-1, 1),
                       x1_loss.view(-1, 1),
                       x2_loss.view(-1, 1),
                       x3_loss.view(-1, 1),
                       x4_loss.view(-1, 1),
                       initial_loss.view(-1, 1),
                       final_loss.view(-1, 1),
                       ]),
            torch.cat([
                net_cost.view(-1, 1),
                       ]),
            )