import time

import numpy as np
from powerbox.dft import fft
from sklearn.neighbors import NearestNeighbors
from spore.model.source_counts import PowerLawSourceCounts

from .c_wrapper import get_visibilities


def knn(baselines, gridpoints, radius=None, k=100, nbrs=None):
    """
    Find either k nearest neighbours, or all neighbours within radius.

    Parameters
    ----------
    baselines : (Nbl, Ndim)-array
        The baseline layout.

    gridpoints : (Ngrid, Ndim)-array
        The gridpoints at which to find the weights

    radius : float, optional
        The radius within which to use baselines

    k : int, optional
        The number of baselines to use for each gridpoint. Not used if radius supplied.

    nbrs : :class:`NearestNeighbors` instance, optional
        An instance of the nearest neighbours which has been pre-fit.

    Returns
    -------
    distances : (Ngrid, k)-array or (Ngrid)-array of 1D arrays.
        The distances to each baseline from each grid point. If `radius` supplied, the result is a
        1D array, each entry of which is a 1D array of variable length (the number of points within the radius).

    indices : same shape as `distances`
        The indices of the baselines included for each grid point.

    nbrs : :class:`NearestNeighbors` instance
        The fitted instance, allowing to pass it back for a future calculation.

    """
    if nbrs is None:
        nbrs = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='ball_tree').fit(baselines)

    if radius is None:
        distances, indices = nbrs.kneighbors(gridpoints)
    else:
        distances, indices = nbrs.radius_neighbors(gridpoints)

    return distances, indices, nbrs


def fourier_beam(q, sigma, renorm=0):
    return np.exp(-sigma ** 2 * (q ** 2 - renorm ** 2))


def generate_spherical_uniform_sample(n, maxphi=np.pi / 2):
    """
    Generate a uniform sampling of the surface of a sphere, down to some azimulthal angle.

    Parameters
    ----------
    n : int
        The size of the sample.
    maxphi : float
        The azimuthal angle down to which to sample. Default, the horizon.

    Returns
    -------
    theta, phi : The circumpolar and azimuthal angles, in radians, of each sample point.

    Notes
    -----
    Sampling function taken from http://mathworld.wolfram.com/SpherePointPicking.html
    """
    if maxphi < 0 or maxphi > np.pi:
        raise ValueError("maxphi needs to be in the range (0,pi)")

    u = np.random.uniform(0, 1, size=n)
    v = np.random.uniform(0, 1, size=n)

    theta = 2 * np.pi * u

    minv = np.cos(maxphi)

    phi = np.arccos((1 + minv) * v - minv)
    return theta, phi


def convert_angles_to_lm(theta, phi):
    stheta = np.sin(theta)
    return stheta * np.cos(phi), stheta * np.sin(phi)


def make_visibilities(u0, f, sigma0, seed=None, threshold=1e-8, account_for=0.95, moment=1, nthreads=1, Smax=1,
                      beta=1.59):
    """
    For some kind of sky model, create a set of visibilities at baselines.

    Parameters
    ----------
    u0 (2,N) array
        Co-ordinates of the baselines
    f :(Nf)-array
        Normalised frequencies of the observation.
    sigma0 : float
        Beam width.
    seed : int, optional
        An optional seed to use in creating the sky.

    Returns
    -------
    vis : (Nf, N)-array
        Visibilities at each frequency and baseline.

    """

    if seed:
        np.random.seed(seed)

    # Assume we peel to 1 Jy.
    Smin = Smax * (1 - account_for)**(1/(moment + 1 - beta))
    sc = PowerLawSourceCounts(Smax, Smin, f, 0, alpha=4100.0, beta=beta)

    nbar = sc.total_number_density

    if hasattr(nbar, "value"):
        nbar = nbar.value

    N = np.random.poisson(nbar * 2 * np.pi)  # Fill the sky

    theta, phi = generate_spherical_uniform_sample(N)

    l, m = convert_angles_to_lm(theta, phi)

    source_flux = sc.sample_source_counts(N)

    print(f" for {N} sources and {len(u0[0])} baselines... (estimated time is {len(f)*len(u0[0])*N*3/1e10} minutes)")

    source_pos = np.array([l, m])

    source_pos_mag = l ** 2 + m ** 2
    source_flux *= np.exp(-source_pos_mag / (sigma0 ** 2))  # by beam

    # Cut down the number of sources to be those who actually contribute
    source_pos = source_pos[:, source_flux >= np.mean(source_flux) * threshold]
    source_flux = source_flux[source_flux >= np.mean(source_flux) * threshold]

    vis = get_visibilities(f, u0.T, source_flux, source_pos.T, nthreads=nthreads)

    return vis


def grid_visibilities_polar(*, u0, vis, f, u, theta, extent, sigma):
    """
    Grid visibilities, with baselines u0, onto a polar grid.

    Parameters
    ----------
    u0 : (2,Nbl) array
        The floating baselines at reference frequency (f=1)
    vis : (Nf,Nbl) array
        The visibilities at the baselines for each frequency.
    f : (Nf)-array
        The normalised frequencies of the observation.
    u : (nu)-array
        The values of u (at reference frequency) at which to evaluate the gridded visibilities.
    theta : (ntheta)-array
        The values of theta to evaluate the gridded visibilities
    extent : float
        Affects the extent to which baselines are tracked to contribute to a grid point.
    sigma : float
        Beam width.

    Returns
    -------
    gridded_vis : (Nf, Nu, Ntheta)-array
        array of visibilities at grid-points, at each frequency.
    """

    if len(vis) != len(f):
        raise ValueError("visibilities has to have the same length as f")

    # Determine X,Y coords of grid points
    U0 = np.outer(u, np.cos(theta)).flatten()
    U1 = np.outer(u, np.sin(theta)).flatten()

    gridpoints = np.array([U0, U1])

    wnut = np.zeros((len(f), len(u), len(theta)))
    wnut2 = np.zeros((len(f), len(u), len(theta)))
    gridded_vis = np.zeros((len(f), len(u), len(theta)), dtype=np.complex128)


    nbrs = None
    for i, visi in enumerate(vis):  # Runs over frequencies
        # dist, ind are 1D arrays of objects, where each object is a 1D array of points within
        # the radius around each grid point.
        distances, indices, nbrs = knn(u0.T, gridpoints.T / f[i],
                                       radius=extent / (2 * sigma),
                                       nbrs=nbrs)

        # we use the fact that (u - fu_i)^2 = f^2 (u/f - u_i)^2 so we don't have to refit baselines everytime.
        distances *= f[i]

        # Calculate weights for each baseline at each point
        weights = np.array([fourier_beam(d, sigma=sigma) for d in distances])

        # Get total weight for each point.
        wnut[i] = np.reshape([np.sum(w) for w in weights], (len(u), len(theta)))
        wnut2[i] = np.reshape([np.sum(w ** 2) for w in weights], (len(u), len(theta)))

        mask = wnut[i] > 0
        gridded_vis[i][mask] = np.reshape(np.array([np.sum(w * visi[ind]) for w, ind in zip(weights, indices)]),
                                          (len(u), len(theta)))[mask] / wnut[i][mask]

    gridded_vis[np.isnan(gridded_vis)] = 0.0
    gridded_vis[np.isinf(gridded_vis)] = 0.0

    return gridded_vis, wnut, wnut2


def power_from_vis(vis, f, taper=None):
    """
    Get the power spectrum at a bunch of vectors u, from the (gridded) visibilities (as functions of frequency)
    there.

    Parameters
    ----------
    vis : (Nf,N)-array
        Array of N visibilities at Nf frequencies.
    f : (Nf)-array
        Frequencies of observation, normalised to reference frequency.
    taper : (Nf)-array or None
        A frequency taper to apply

    Returns
    -------
    power : (Nf/2-1)-array
        The power spectrum at each visibility
    omega : (Nf/2-1)-array
        The fourier-dual of frequency corresponding to `f`. Note that these correspond to `f` *not* `nu`.
        To yield a power spectrum which corresponds to nu, multiply the power by nu0**2 and eta by 1/nu0
    """
    if taper is None:
        taper = 1

    res, omega = fft((taper * vis.T).T, L=(f[-1] - f[0]), axes=(0,), b=2 * np.pi, a=0)
    omega = omega[0]

    res = np.abs(res) ** 2
    res = res[len(f) // 2 + 1:]
    omega = omega[len(f) // 2 + 1:]
    return res, omega


def generate_3d_power(u0, f, sigma0, u, theta, taper=None, extent=50, nthreads=1):
    if taper is None:
        taper = 1

    t = time.time()
    print("Generating Visibilities...", end="")
    vis = make_visibilities(u0, f, sigma0, nthreads=nthreads)
    t1 = time.time()
    print(f"   ... took {(t1-t)/60} minutes.")

    print("Gridding Visibilities...")
    vis, weights, weights2 = grid_visibilities_polar(u0=u0, vis=vis, f=f, u=u, theta=theta, sigma=sigma0, extent=extent)
    t2 = time.time()
    print(f"   ... took {t2-t1} seconds.")

    # Weights are fn of nu, need to get sum over nu
    weights = np.nansum(weights2.T * taper(len(f)) / weights.T ** 2, axis=-1).T

    print("Generating Power...")
    # This gets the power at each grid-point (so 3D power)
    power, omega = power_from_vis(vis, f, taper(len(f)))
    t3 = time.time()
    print(f"   ... took {t3-t2} seconds.")

    return power, weights, omega


def generate_2d_power(u0, f, sigma, taper=None, umin=None, umax=None, nu=100, ntheta=50, extent=50, nthreads=1):

    if umax is None:
        umax = np.sqrt(np.max(u0[0] ** 2 + u0[1] ** 2)) / 1.2
    if umin is None:
        umin = np.sqrt(np.min(u0[0] ** 2 + u0[1] ** 2)) * 1.2

    # Setup grid
    dtheta = 2 * np.pi / ntheta

    u = np.logspace(np.log10(umin), np.log10(umax), nu)
    theta = np.arange(ntheta) * dtheta

    # Get power at each grid point.
    power, weights, omega = generate_3d_power(
        u0=u0, f=f, sigma0=sigma, u=u, theta=theta,
        taper=taper,
        extent=extent, nthreads=nthreads
    )

    wtot = np.sum(weights, axis=-1)
    power = np.sum(power * weights, axis=-1) / wtot
    power[np.isnan(power)] = 0.0

    return power, omega, wtot


def generate_2d_power_sparse(f, sigma, umin, umax, taper=None, nu = 100, ntheta = 50, extent=50, nthreads=1):
    # Setup grid
    dtheta = 2 * np.pi / ntheta

    u = np.logspace(np.log10(umin), np.log10(umax), nu)
    theta = np.arange(ntheta) * dtheta

    power = [0]*nu
    for iu, uu in enumerate(u):
        # Get a bunch of u0 at this u circle.
        u0 = np.vstack((uu * np.cos(theta), uu * np.sin(theta)))

        # Get power at each grid point.
        power[iu], _, omega = generate_3d_power(
            u0=u0, f=f, sigma0=sigma, u=np.array([uu]),
            theta=theta, taper=taper,extent=extent, nthreads=nthreads
        )

    power = np.array(power)
    power = np.mean(power, axis=-1)
    power[np.isnan(power)] = 0.0

    return np.transpose(power, (2, 1, 0)), omega