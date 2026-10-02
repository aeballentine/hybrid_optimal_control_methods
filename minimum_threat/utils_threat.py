from torch import nn
from collections import OrderedDict
import torch
import pandas as pd
import os


class NeuralNetwork(nn.Module):
    def __init__(self, dimensions, function=None, output_activation=None):
        super().__init__()

        stacking = OrderedDict()
        i = 0
        while len(dimensions) > 2:
            stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
            if function:
                stacking["act" + str(i)] = function
            dimensions.pop(0)
            i += 1
        stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
        if output_activation is not None:
            stacking["act" + str(i)] = output_activation
        self.linear_stack = nn.Sequential(stacking)

    def forward(self, inp):
        logits = self.linear_stack(inp)
        return logits.view(-1, 4, 4)


# =====================


def threat(x1: torch.tensor, x2: torch.tensor, t: torch.tensor, device):
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
    coeff_peaks = torch.tensor(
        # [0.77, 0.6, 0.71, 0.98, 0.87, 0.85, 0.86, 0.81, 0.74],  # intensity
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # intensity
         [-13.86, -3.43, 11.3, -11.71, -2.42, 11.6, -10.76, 4.87, 11.45],  # x location
         [-12.28, -8.35, -10.04, -3.88, 4.29, -13.34, 10.17, 11.58, 8.24],  # y location
         [2.57, 4.92, 4.29, 1.3, 3.7, 3.38, 6.56, 3.21, 2.84],  # x spread
         [3.45, 2.64, 1.76, 2.51, 2.93, 2.72, 2.91, 0.67, 1.26]]).to(device)  # y spread

    const_1 = 1.0  # 1/2.32

    for m11 in range(n_peaks):
        a = coeff_peaks[0, m11] * (1 + 0.6 * torch.sin(0.312 * t + 2 * m11) + 0.4 * torch.cos(1.75 * t + 3 * m11))  # + coeff_peaks[0, m11]

        # drifting centers
        cx = coeff_peaks[1, m11] + 2.0 * (0.2 * torch.sin(t + m11) + 0.3 * torch.sin(1.414 * t + 3 * m11) + 0.5 * torch.cos(1.732 * t + 2 * m11))
        cy = coeff_peaks[2, m11] + 2.0 * (0.6 * torch.sin(t + m11) + 0.1 * torch.sin(1.414 * t + 3 * m11) + 0.3 * torch.cos(1.732 * t + 2 * m11))

        c_xym = torch.exp(
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


def save_res(x0, y0, model_states,
             model_tf, dt, save_path, device, threat_func, training_time, lamb, vel):
    final_time_path = model_tf

    num_colloc = torch.ceil(torch.max(final_time_path) / dt + 1).to(torch.int16).item()
    tau = torch.linspace(0, 1, num_colloc, requires_grad=True).view(-1, 1).to(device)
    x = x0 * torch.ones((num_colloc,), requires_grad=True, dtype=torch.float32).view(-1, 1).to(device)
    y = y0 * torch.ones((num_colloc,), requires_grad=True, dtype=torch.float32).view(-1, 1).to(device)
    training_time = training_time * torch.ones((num_colloc * 4,)).view(-1)
    const_lambda = lamb * torch.ones((num_colloc * 4,)).view(-1)
    const_vel = vel * torch.ones((num_colloc * 4,)).view(-1)

    final_time = model_tf.repeat_interleave(num_colloc).view(-1, 1)
    t_real = final_time * tau.repeat((4, 1))

    states = model_states(tau)

    x1 = states[:, :, 0].T.reshape(-1, 1)
    x2 = states[:, :, 1].T.reshape(-1, 1)
    psi = states[:, :, 2].T.reshape(-1, 1)

    magnitude = states[:, :, 3].T.reshape(-1, 1)#.view(-1, 1)
    p1 = magnitude * torch.cos(psi)
    p2 = magnitude * torch.sin(psi)

    res_dataframe = pd.DataFrame({
        "x10": x.repeat((4, 1)).view(-1).detach().cpu().numpy(),
        "x20": y.repeat((4, 1)).view(-1).detach().cpu().numpy(),
        "training_time": training_time.cpu().numpy(),
        "const_lambda": const_lambda,
        "const_vel": const_vel,
        "x1": x1.view(-1).detach().cpu().numpy(),
        "x2": x2.view(-1).detach().cpu().numpy(),
        "t": t_real.view(-1).detach().cpu().numpy(),
        "psi": psi.view(-1).detach().cpu().numpy(),
        "rho": magnitude.view(-1).detach().cpu().numpy(),
        "p1": p1.view(-1).detach().cpu().numpy(),
        "p2": p2.view(-1).detach().cpu().numpy(),
    })
    os.makedirs(save_path.split('/')[0], exist_ok=True)
    res_dataframe.to_csv(save_path)