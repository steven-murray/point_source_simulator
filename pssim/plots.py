import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm


def plot_power(power, u, omega, min_oom=None, u_is_logscale=True, omega_is_logscale=True,
               wedge=0, colorbar=True, xlabel=True, ylabel=True, ax=None, fig=None, vmin=None, vmax=None,
               cmap=None, cbar_title=None, lognorm=False, horizon_line=5,
               **kwargs):
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1)

    if u_is_logscale:
        umin = np.log10(u[0])
        umax = np.log10(u[-1])
    else:
        umin = u[0]
        umax = u[-1]

    if omega_is_logscale:
        omegamin = np.log10(omega[0])
        omegamax = np.log10(omega[-1])
    else:
        omegamin = omega[0]
        omegamax = omega[-1]

    if vmin is None and vmax is None and min_oom is not None:
        if hasattr(min_oom, "__len__"):
            vmin, vmax = min_oom
        else:
            vmin = power.max() / 10 ** min_oom
            vmax = power.max()

    if vmax is None:
        vmax = power.max()

    posv = power.min() >= 0

    if cmap is None:
        cmap = "viridis" if posv else "bwr"

    im = ax.imshow(
        power, origin='lower',
        norm=(LogNorm() if posv else SymLogNorm(vmax / 1e5)) if lognorm else None,
        extent=(umin, umax, omegamin, omegamax),
        vmin=vmin, vmax=vmax, aspect='auto',
        cmap=cmap,
        **kwargs
    )

    if xlabel:
        ax.set_xlabel(r"$\log_{10} u$", fontsize=13)

    if not u_is_logscale:
        ax.set_xscale('log')
    if not omega_is_logscale:
        ax.set_yscale('log')
        if ylabel:
            ax.set_ylabel(r"$\omega$", fontsize=13)
    elif ylabel:
        ax.set_ylabel(r"$\log_{10}  \omega$", fontsize=13)

    if colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_title or "Power", fontsize=13)
    if wedge > 0:
        ax.contour(np.log10(u), omega, power, wedge)

    if horizon_line:
        ax.plot(np.log10(u) if u_is_logscale else u, np.log10(u) + np.sqrt(np.log(10)*horizon_line) if omega_is_logscale else u*np.sqrt(np.log(10)*horizon_line), color='k')

    return fig

#
# def waterfall(u0, f, u, theta, extent, sigma, ax=None, fig=None, trajec=True):
#     if fig is None:
#         fig, ax = plt.subplots(1, 1)
#
#     weight, weight2 = grid_baselines(u0, f, u, np.array([theta]), extent, sigma)
#
#     print(weight.shape)
#     ax.imshow(weight[:, :, 0], origin='lower', extent=(np.log10(u.min()), np.log10(u.max()), f.min(), f.max()),
#               aspect='auto')
#
#     if trajec:
#         for uu in u[::4]:
#             ax.plot(np.log10(f * uu), f, color='k')
#
#     ax.set_ylim((f.min(), f.max()))
#     ax.set_xlim(np.log10(u.min()), np.log10(u.max()))
#
#     return fig
