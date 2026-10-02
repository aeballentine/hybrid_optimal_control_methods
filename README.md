# Hybrid Machine-Learning (ML)-Numerical Methods for Optimal Control
This gives the author's implementation of the hybrid ML-numerical method for optimal control problems. The method uses a physics-informed neural network (PINN) to predict the inital co-states and final time. This repo presents two case studies: a satellite launch problem and minimum exposure trajectory planning in a spatially-temporally varying scalar field. 

## Sample Problems
### Satellite Launch
Objective: launch a satellite into low-Earth orbit in minimum time, considering a flat Earth model and the effects of drag

$$\min_{t_{f}} \int_0^{t_{f}} \mathrm{d} t,$$
$$\mathrm{s.t.}\: \dot{x}_1(t) = x_3(t),$$
$$\dot{x}_2(t) = x_4(t),$$
$$\dot{x}_3(t) = \frac{T}{m(t)} \cos u(t) - \frac{D(\vec{x}(t))}{m(t)} \cos\gamma,$$
$$\dot{x}_4(t) = \frac{T}{m(t)} \sin u(t) - \frac{D(\vec{x}(t))}{m(t)} \sin\gamma - g,$$
$$x_1(0) = x_2(0) = x_3(0) = x_4(0) = 0,$$
$$x_2(t_\mathrm{f}) = x_\mathrm{2f}, \: x_3(t_\mathrm{f}) = x_\mathrm{3f}, \: x_4(t_\mathrm{f}) = x_\mathrm{4f},$$

### Minimum Exposure Navigation in a Spatiotemporally Varying Scalar Field
$$\min_{t_\mathrm{f}} \int_0^{t_\mathrm{f}} c(\vec{x}(t), t) \mathrm{d} t,$$
$$\mathrm{s.t.}\: \dot{x}_1(t) = v \cos\psi(t),$$
$$\dot{x}_2(t) = v \sin\psi(t),$$
$$\vec{x}(0)=\vec{x}_0,$$
$$\vec{x}(t_\mathrm{f})=\vec{0}.$$

## Overview of the Code
