from functools import partial
from os import path

import click
import numpy as np

from pssim import layouts
from pssim.mapping import numerical_power_vec, numerical_sparse_power_vec


def gaussian_taper(tau, fmax, n):
    f = np.linspace(1 - (fmax - 1), fmax, n)

    return np.exp(-(f - 1) ** 2 * tau ** 2)


@click.command()
@click.argument("kind", type=click.Choice(["circle", "filled_circle", "spokes", "rlx_boundary", "rlx_grid", "hexagon",
                                           'spokes-pure', 'sparse']))
@click.option("-N", "--n_antenna", default=128, type=int, help="(approximate) number of antennae in layout")
@click.option("-p", '--prefix', default="")
@click.option("-P", "--plot/--no-plot", default=False)
@click.option("-l", "--log/--linear", default=True)
@click.option("--shape", type=float, default=1.0, help="density gradient of antennas for filled shapes")
@click.option("-n", "--nspokes", type=int, default=1, help='number of spokes in relevant layouts')
@click.option("-u", "--umin", type=float, default=30.0, help='minimum u')
@click.option("-U", "--umax", type=float, default=800.0, help='maximum u')
@click.option("--ugrid-size", type=int, default=20, help="number of points on the u-grid")
@click.option("-w", "--omega-min", type=float, default=40.0, help='minimum omega')
@click.option("-W", "--omega-max", type=float, default=500.0, help='maximum omega')
@click.option("--omega-grid-size", type=int, default=20, help="number of points on the omega-grid")
@click.option("-m", "--u0-min", type=float, default=30.0, help='minimum u0 (if applicable')
@click.option("-j", "--processes", type=int, default=1, help='number of processes to use')
@click.option("-t", "--threads", type=int, default=1, help='number of threads to use')
@click.option("-v", "--frequency", type=float, default=150.0, help='central frequency of observation (MHz)')
@click.option("-T", "--tau", type=float, default=100.0, help='band precision')
@click.option("--taper", type=str, default=None, help="kind of taper (must be available in numpy)")
@click.option("-h", "--threshold", type=float, default=5.0, help='order-of-magnitude threshold tolerance')
@click.option("-d", "--diameter", type=float, default=4.0, help='antenna (tile) diameter')
@click.option("-v", "--verbose/--not-verbose", default=False)
@click.option('--restart/--continue', default=False,
              help='whether to restart the integration, or continue if found.')
@click.option("-r", "--realisations", type=int, default=200, help="how many realisations to run if doing numerical.")
@click.option("--bw", type=float, default=10.0, help='bandwidth (MHz) (if numerical)')
@click.option("--sky-moment", type=float, default=1, help="moment of source counts for which to calculate threshold")
@click.option("--smax", type=float, default=1, help="flux density (Jy) of upper limit of source counts (peeling limit)")
@click.option("--large/--not-large", default=False, help="make log-spoke models expand indefinitely")
def main(kind, n_antenna, prefix, plot, log, shape, nspokes, umin, umax, ugrid_size, omega_min, omega_max,
         omega_grid_size,
         u0_min, processes, threads, frequency, tau, taper, threshold, diameter, verbose, restart, realisations, bw,
         sky_moment, smax, large):
    """
    Run numerically simulated power spectra from point-source only skies.
    """
    numerical = True

    input_args = locals()

    sigma = 0.42 * 3e2 / (frequency * diameter)

    # Now pad the diameter to make sure no "square" tiles overlap:
    # diameter *= np.sqrt(2)

    u = np.logspace(np.log10(umin), np.log10(umax), ugrid_size)
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), omega_grid_size)

    name = kind
    extras = (f"{n_antenna}_{umin:.2f}_{umax:.2f}_{ugrid_size}_{omega_grid_size}_{omega_min:.2f}_{omega_max:.2f}" +
              f"_{frequency:.0f}_{tau if taper is None else taper}_{threshold:.0f}_{diameter:.1f}_{sky_moment}_{smax}{'_large' if large else ''}")

    if kind == "circle":
        u0, x = layouts.get_baselines_circle(n_antenna, umax=umax, antenna_diameter=diameter)
    elif kind == "filled_circle":
        u0, x = layouts.get_baselines_filled_circle(n_antenna, umax=umax, alpha=-shape,
                                                    antenna_diameter=diameter)
        name += "_%s" % shape
    elif kind == 'spokes':
        u0, x = layouts.get_baselines_spokes(n_antenna, umax=umax, nspokes=nspokes, umin=u0_min,
                                             log=log, antenna_diameter=diameter, large=large)
        name += "_%s_%s_%.1f" % ('log' if log else "lin", nspokes, u0_min)
    elif kind == 'rlx_boundary':
        u0, x = layouts.get_baselines_rlx_boundary(n_antenna, umax=umax, antenna_diameter=diameter)
    elif kind == 'rlx_grid':
        u0, x = layouts.get_baselines_rlx_grid(n_antenna, umax=umax, nr=10, umin=u0_min, log=log,
                                               antenna_diameter=diameter)
        name += "_%s_%.1f" % ('log' if log else 'lin', u0_min)
    elif kind == 'hexagon':
        u0, x = layouts.get_baselines_hexagon(n_antenna, umax=umax, antenna_diameter=diameter)
    elif kind == "spokes-pure":
        fmax = 1 + threshold / np.sqrt(2) / tau
        d = np.sqrt(threshold * np.log(10) / (2 * np.pi ** 2 * sigma ** 2))

        if shape == 1 or shape > 2:
            # Derived Extents.

            n_per_spoke = int(np.log(umax / umin) / np.log(fmax + d / umin)) + 1
            umax = (fmax + d / umin) ** (n_per_spoke - 1) * umin
            name += "_sbl"
        elif shape == 2:
            f = ((umin - d) + np.sqrt((umin - d) ** 2 + 8 * d * umin)) / (2 * umin)
            alpha = f / (2 - f)

            n_per_spoke = int(np.log(umax / umin) / np.log(alpha)) + 1
            umax = (fmax + d / umin) ** (n_per_spoke - 1) * umin
            name += "_sblpf"

        u0 = layouts.get_concentric_baselines(umin, umax, n_per_spoke, nspokes, log=True)
        u = np.logspace(np.log10(umin), np.log10(umax), n_per_spoke)

        if shape > 2:
            name += "_%.2f" % shape
            # Make redundant baselines.
            new_ind = np.random.random_integers(int(u0.shape[1] / 1.5), int(u0.shape[1] / 1.5) + 4,
                                                size=int((shape - 2) * u0.shape[1]))
            u0 = np.hstack((u0, u0[:, new_ind]))

    name = "numerical_" + name
    fname = path.join(prefix, name + extras + '.h5')

    f = np.linspace(frequency - bw / 2, frequency + bw / 2, omega_grid_size * 2 + 1) / frequency

    if taper is not None:
        taper = getattr(np, taper)
    else:
        taper = partial(gaussian_taper, tau, f)

    if kind == "sparse":
        numerical_sparse_power_vec(
            fname=fname, umin=umin, umax=umax, nu=ugrid_size,
            taper=taper, sigma=sigma, f=f, realisations=realisations,
            nthreads=threads, restart=restart, extent=threshold, processes=processes, sky_moment=sky_moment, Smax=smax
        )
    else:
        numerical_power_vec(
            fname=fname, u0=u0, umin=umin, umax=umax, nu=ugrid_size, taper=taper, sigma=sigma, f=f,
            realisations=realisations, nthreads=threads, restart=restart, extent=threshold, processes=processes,
            sky_moment=sky_moment, Smax=smax
        )
