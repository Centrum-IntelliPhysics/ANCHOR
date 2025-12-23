#!/usr/bin/env python
# coding: utf-8

import jax
# Enable double precision
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
from tqdm import trange

# =========================================
# Configuration & Parameters
# =========================================
N = 32          # grid points per dimension
L = 1.0         # domain size
eps = 0.05      # interface width parameter


# =========================================
# Spectral grid setup
# =========================================
kx = 2 * jnp.pi * jnp.fft.fftfreq(N, d=L / N)
KX, KY = jnp.meshgrid(kx, kx, indexing='ij')
K2 = KX**2 + KY**2
L_op = -eps**2 * K2

# =========================================
# Initial condition: Smoothed random field
# =========================================
def smooth_grf(key, cutoff_ratio=0.05):
    """
    Gaussian‑filtered random field, linearly mapped to [-1,1]
    so that it stays perfectly smooth (no clipping kinks).
    """
    # white noise
    noise     = jax.random.normal(key, (N, N))
    noise_hat = jnp.fft.fft2(noise)

    # Gaussian low‑pass filter: correlation length ℓ = cutoff_ratio * L
    ℓ           = cutoff_ratio * L
    gauss_filt  = jnp.exp(-0.5 * K2 * ℓ**2)

    # apply in spectral space, back to real
    field = jnp.fft.ifft2(noise_hat * gauss_filt).real

    # pure linear map to [-1,1]
    fmin, fmax = field.min(), field.max()
    return 2 * (field - fmin) / jnp.maximum(fmax - fmin, 1e-12) - 1.0


# =========================================
# ETDRK4 time stepping
# =========================================
@jax.jit
def etdrk4_step(u, h):
    expL   = jnp.exp(L_op * h)
    expL2  = jnp.exp(L_op * h/2)
    Lh     = L_op * h
    small  = jnp.abs(Lh) < 1e-6
    phi1   = jnp.where(small, h,       (expL - 1)       / L_op)
    phi2   = jnp.where(small, h**2/2,   (expL - 1 - Lh)  / L_op**2)
    phi3   = jnp.where(small, h**3/6,   (expL - 1 - Lh - 0.5*Lh**2) / L_op**3)

    def N(u): return u**3 - u

    u_hat = jnp.fft.fft2(u)
    N1    = jnp.fft.fft2(-N(u))

    a_hat = expL2 * u_hat + phi1 * N1 / 2
    ua    = jnp.fft.ifft2(a_hat).real
    N2    = jnp.fft.fft2(-N(ua))

    b_hat = expL2 * u_hat + phi1 * N2 / 2
    ub    = jnp.fft.ifft2(b_hat).real
    N3    = jnp.fft.fft2(-N(ub))

    c_hat = expL2 * a_hat + phi1 * (N3 - N1) / 2
    uc    = jnp.fft.ifft2(c_hat).real
    N4    = jnp.fft.fft2(-N(uc))

    u_new_hat = (
        expL * u_hat
        + phi1 * N1
        + 2 * phi2 * (N2 + N3)
        + phi3 * N4
    )
    return jnp.fft.ifft2(u_new_hat).real

# =========================================
# PDE time derivative & energy
# =========================================
@jax.jit
def compute_ut(u):
    u_hat = jnp.fft.fft2(u)
    lap   = jnp.fft.ifft2(-K2 * u_hat).real
    return eps**2 * lap - (u**3 - u)

def compute_energy(u_snaps):
    """
    Real‐space energy:
      E = ∫[ eps²/2 |∇u|² + 1/4 (u²−1)² ] dx
    approximated with central differences and trapezoidal rule.
    """
    energies = []
    dx = L / N

    for u in u_snaps:
        # periodic roll for derivatives
        ux = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2*dx)
        uy = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2*dx)

        grad_term = 0.5 * eps**2 * jnp.mean(ux**2 + uy**2)
        pot_term  = jnp.mean(0.25 * (u**2 - 1)**2)

        energies.append(grad_term + pot_term)
    return jnp.array(energies)


def finite_difference_ut(u_array, dt):
    u  = u_array
    ut = np.zeros_like(u)
    ut[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    ut[0]    = (u[1] - u[0])   / dt
    ut[-1]   = (u[-1] - u[-2]) / dt
    return ut

# =========================================
# Solver function
# =========================================
def solve_allencahn(u,T_final):
    # u = smooth_grf(key)
    # T_final = 1.0   # final time
    dt = 0.01       # coarse time step
    refine = 200   # refine factor for ETDRK4
    dt_fine = dt / refine
    n_steps_fine = int(T_final / dt_fine)
    n_coarse = int(T_final / dt) + 1
    u_record  = jnp.zeros((n_coarse, N, N))
    ut_record = jnp.zeros((n_coarse, N, N))
    u_record  = u_record.at[0].set(u)
    ut_record = ut_record.at[0].set(compute_ut(u))

    idx = 1
    for step in range(1, n_steps_fine + 1):
        u = etdrk4_step(u, dt_fine)
        if step % refine == 0 or step == n_steps_fine:
            u_record  = u_record.at[idx].set(u)
            ut_record = ut_record.at[idx].set(compute_ut(u))
            idx += 1

    # assert idx == n_coarse, f"Snapshots recorded {idx}, expected {n_coarse}"
    energies = compute_energy(u_record)
    return u_record, ut_record, energies

# =========================================
# Plotting & diagnostics
# =========================================
def plot_diagnostics(u_rec, ut_rec, energies):
    times = np.linspace(0, T_final, n_coarse)
    print(f"Start energy: {energies[0]:.6f}, End energy: {energies[-1]:.6f}, Change: {energies[-1]-energies[0]:.2e}")

    idxs = np.linspace(0, n_coarse-1, 5, dtype=int)
    u_np  = np.array(u_rec)
    ut_np = np.array(ut_rec)
    ut_fd = finite_difference_ut(u_np, dt)
    diff  = ut_np - ut_fd

    lim_u     = np.max(np.abs(u_np))
    lim_ut    = np.max(np.abs(ut_np))
    lim_fd    = np.max(np.abs(ut_fd))
    lim_diff  = np.max(np.abs(diff))

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(4, len(idxs), wspace=0.05, hspace=0.1, 
                           left=0.05, right=0.9, bottom=0.05, top=0.95)
    row_labels = ['u', '∂u/∂t (ETD)', '∂u/∂t (FD)', 'Difference']
    cmaps = ['RdBu', 'PuOr', 'PuOr', 'bwr']
    vmins = [-lim_u, -lim_ut, -lim_fd, -lim_diff]
    vmaxs = [ lim_u,  lim_ut,  lim_fd,  lim_diff]

    for i, label in enumerate(row_labels):
        for j, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[i, j])
            data = [u_np[idx], ut_np[idx], ut_fd[idx], diff[idx]][i]
            im = ax.imshow(data, cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i], extent=[0, L, 0, L])
            ax.set_title(f"{label}, t={times[idx]:.2f}", fontsize=10)
            ax.axis('off')

    for i in range(4):
        cax = fig.add_axes([0.91, 0.95 - (i+1)*0.23, 0.02, 0.18])
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps[i],
                                           norm=plt.Normalize(vmin=vmins[i], vmax=vmaxs[i])),
                     cax=cax, label=row_labels[i])

    fig.suptitle("Allen–Cahn 2D Diagnostics — Gridspec Visualization & Improvements", fontsize=18)
    plt.show()




