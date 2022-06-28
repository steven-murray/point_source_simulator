import warnings

import numpy as np

from . import c_wrapper as cwrap


def bl_from_x(x, y, antenna_diameter=4.0, exception=True, ignore_redundant=True):
    res = cwrap.get_baselines(x, y, antenna_diameter)

    if np.isscalar(res):
        # didn't work.
        if exception:
            raise RuntimeError("Some of the antennas were too close together")

        else:
            x, y = cwrap.get_good_antennas(x, y, antenna_diameter, -res)
            return x, y
    else:
        if ignore_redundant:
            np.round(res, 3)
            res = np.unique(res, axis=-1)

        return res / 2


def _bl_from_x(x, y, antenna_diameter=4.0, exception=True, ignore_redundant=True):
    "Gives *all* the distances between antennae, in units of wavelengths, at lambda = 2m (150 MHz)"

    dist = [np.add.outer(x, -x).flatten(), np.add.outer(y, -y).flatten()]
    np.round(dist, 3)

    if ignore_redundant:
        dist = np.unique(dist, axis=-1)

    # Remove autocorrelations
    dist_mag2 = dist[0] ** 2 + dist[1] ** 2
    dist = dist[:, dist_mag2 > 0]
    dist_mag2 = dist_mag2[dist_mag2 > 0]

    if np.any(dist_mag2 < antenna_diameter ** 2):
        if exception:
            raise RuntimeError(
                "%s of the baselines were smaller than the antenna diameter (min=%s, diam=%s)" % (
                    np.sum(dist_mag2 < antenna_diameter ** 2),
                    np.sqrt(dist_mag2.min()),
                    antenna_diameter
                )
            )
        else:
            warnings.warn(
                "%s of the baselines were smaller than the antenna diameter (min=%s, diam=%s)" % (
                    np.sum(dist_mag2 < antenna_diameter ** 2),
                    np.sqrt(dist_mag2.min()),
                    antenna_diameter
                )
            )

    return dist / 2  # divide by two, which is c/nu with nu=150MHz


def get_baselines_circle(N, umax, antenna_diameter=4.0):
    xmax = umax * 2

    if np.pi * xmax / N < antenna_diameter:
        raise Exception("Can't physically fit those antennae!")

    theta = np.arange(N) * 2 * np.pi / N
    x = xmax * np.cos(theta) / 2  # /2 since xmax is a diameter, not radius.
    y = xmax * np.sin(theta) / 2

    return bl_from_x(x, y, antenna_diameter), [x, y]


def get_baselines_filled_circle(N, umax, alpha=0, antenna_diameter=4.0, level=0, start_x=None, start_y=None,
                                maxlevel=100):
    "alpha=0 corresponds to uniform disk, -1 to log disk"

    if level > maxlevel:
        raise ValueError("couldn't get antennas fit into the circle in %s attempts." % maxlevel)

    if start_x is None:
        start_x = np.array([])
    if start_y is None:
        start_y = np.array([])

    assert len(start_x) == len(start_y)

    n = N - len(start_y)

    print("LEVEL=%s, N=%s, n=%s" % (level, N, n))

    xmax = umax * 2.

    if N * antenna_diameter ** 2 > np.pi * xmax ** 2:
        raise Exception("Can't physically fit those antennae!")

    theta = np.random.uniform(0, 2 * np.pi, size=n)
    r = np.random.uniform(0, 1, size=n) ** (1 / (2. + alpha)) * xmax

    x = r * np.cos(theta) / 2  # /2 since xmax is a diameter, not radius
    y = r * np.sin(theta) / 2

    x = np.concatenate((start_x, x))
    y = np.concatenate((start_y, y))

    u = bl_from_x(x, y, antenna_diameter, exception=False)

    if not hasattr(u, "shape"):
        # Try again by replacing some of the antennas.
        u, [x, y] = get_baselines_filled_circle(N, umax, alpha, antenna_diameter,
                                                level=level + 1, start_x=u[0], start_y=u[1], maxlevel=maxlevel)

    return u, [x, y]


def _get_baselines_spokes_linear(N, radius, n_per_spoke, nspokes, antenna_diameter=4.0):
    """
    Generate an antenna layout that is very close to a completely regular concentric grid.

    In order to fit baselines that are close to the centre, successively remove more from the centre rings until
    they fit.

    Parameters
    ----------
    N : int
        Approximate (max) number of antennas in the array.
    nspokes :
        The number of spokes in the layout (at largest radius). base**nspokes
    umax : float
        Defines the maximum radius of the array. Defined to be atw 150 MHz.
    antenna_diameter : float, optional
        The diameter of the antennae, assumed to be circular. Used to determine if they "fit".
    base : int
        When antennas are iteratively removed to make them fit, they are removed on this basis.

    Returns
    -------

    """
    if radius / n_per_spoke < antenna_diameter:
        try_Ns = np.ceil(antenna_diameter * (N - 1) / radius)
        try_N = int(nspokes * radius / antenna_diameter + 1)
        try_rad = antenna_diameter * n_per_spoke
        raise Exception(
            f"Can't physically fit those antennae (radially)! Try using at least {try_Ns} spokes or less than {try_N} antennae or umax at least {try_rad}")

    r = np.linspace(0, radius, n_per_spoke + 1)[1:]

    return r


def _get_baselines_spokes_log(n_per_spoke, radius, umin=None, antenna_diameter=4.0):
    """
    Create a spoke layout which attempts to fit as many baselines in an "almost regular" pattern as possible
    """

    inner_radius = umin

    if inner_radius < antenna_diameter:
        raise Exception("Can't physically fit those antennae (inner radius is too small)!")

    r = np.logspace(np.log10(inner_radius), np.log10(radius), n_per_spoke)

    if r[1] - r[0] < antenna_diameter:
        raise Exception("Can't physically fit those antennae (radially)! %s" % (r[1] - r[0]))

    return r


def _get_baselines_spokes_log_large(n_per_spoke, umin, umax, antenna_diameter=4.0):
    """
    Create a spoke layout which attempts to fit as many baselines in an "almost regular" pattern as possible
    """

    inner_radius = umin

    if inner_radius < antenna_diameter:
        raise Exception("Can't physically fit those antennae (inner radius is too small)!")

    delta = (umax/umin)**(1./n_per_spoke)

    r = np.array([umin * np.sum(delta**(np.arange(i)-1)) for i in range(1,n_per_spoke+1)])

    return r


def get_baselines_spokes(N, nspokes, umax, umin=None, log=True, antenna_diameter=4.0, base=3, large=False):
    nspokes = base ** nspokes
    n_per_spoke = int((N - 1) / nspokes)  # number of antennas in each spoke (1 in the middle in every spoke).

    radius = umax

    if umin is None:
        umin = umax / 100

    if log:
        if large:
            r = _get_baselines_spokes_log_large(n_per_spoke, umin, umax, antenna_diameter)
        else:
            r = _get_baselines_spokes_log(n_per_spoke, radius, umin, antenna_diameter)

    else:
        r = _get_baselines_spokes_linear(N, radius, n_per_spoke, nspokes, antenna_diameter)

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / nspokes)

    R = [[rr] * len(theta) for rr in r]
    THETA = [theta] * len(r)

    # Now go through each r and see how many can fit
    for i, rr in enumerate(r):
        circumference = 2 * np.pi * rr
        d = circumference / nspokes

        if d < antenna_diameter:
            nn = int(np.ceil(np.log(antenna_diameter / d) / np.log(base)))
            every = base ** nn

            R[i] = R[i][::every]
            THETA[i] = THETA[i][::every]

    R = [0] + sum(R, [])
    THETA = np.concatenate([[0], np.concatenate(THETA)])  # flatten the lists

    if len(R) < n_per_spoke * nspokes + 1:
        warnings.warn(
            "From an attempt of laying %s antennae, only %s could be laid" % (n_per_spoke * nspokes + 1, len(R)))

    R = np.array(R)
    THETA = np.array(THETA)

    x = R * np.cos(THETA)
    y = R * np.sin(THETA)

    return bl_from_x(x, y, antenna_diameter), [x, y]


def get_baselines_rlx_boundary(N, umax, antenna_diameter=4.0):
    xmax = umax * 2

    if np.pi * xmax / N < antenna_diameter:
        raise Exception("Can't physically fit those antennae! %s<%s" % (umax / N, antenna_diameter))

    theta1 = np.linspace(0, np.pi / 3, N / 3 + 1)
    x1 = xmax * np.cos(theta1) - xmax / 2
    y1 = xmax * np.sin(theta1) - xmax * (np.sin(np.pi / 3) - 0.5)

    theta2 = np.linspace(2 * np.pi / 3., np.pi, N / 3 + 1)[1:]
    x2 = xmax * np.cos(theta2) + xmax / 2
    y2 = xmax * np.sin(theta2) - xmax * (np.sin(np.pi / 3) - 0.5)

    theta3 = np.linspace(-2 * np.pi / 3, -np.pi / 3, N / 3 + 1)[1:-1]
    x3 = xmax * np.cos(theta3)
    y3 = xmax * np.sin(theta3) + xmax / 2

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    return bl_from_x(x, y, antenna_diameter), [x, y]


def get_baselines_rlx_grid(N, umax, nr, umin=None, log=True, antenna_diameter=4.0):
    xmax = umax * 2

    if xmax / nr / 2 < antenna_diameter:
        raise Exception("Can't physically fit those antennae!")

    if log:
        if umin is None:
            umin = umax / 100

        xmin = umin * 2
        r = np.logspace(np.log10(umin), np.log10(umax), nr)
    else:
        if umin is None:
            umin = 0
        xmin = umin * 2
        r = np.linspace(umin, umax, nr)

    x = [0] * nr
    for i, rr in enumerate(r):
        _, x[i] = get_baselines_rlx_boundary(N // nr, rr)
        x[i] = np.array(x[i])

    x = np.transpose(np.array(x), (1, 0, 2)).reshape((2, -1))
    return bl_from_x(x[0], x[1], antenna_diameter), x


def get_baselines_hexagon(N, umax, antenna_diameter=4.0):
    n = int(0.5 + 0.5 * np.sqrt(1 + 4 * (N - 1) / 3.))

    xmax = umax * 2

    if xmax / n < antenna_diameter:
        raise Exception("Can't fit the antennae!")

    x = np.zeros(3 * n ** 2 - 3 * n + 1)
    y = np.zeros(3 * n ** 2 - 3 * n + 1)
    k = 0
    for i in range(n, n + n):
        for j in range(i):
            x[k] = j - i * 0.5
            y[k] = 2 * n - 1 - i
            k += 1

    # Flip over the top half
    print(n, N, k, 3 * n ** 2 - 3 * n + 1)
    x[k:] = x[:k - 2 * n + 1]
    y[k:] = -y[:k - 2 * n + 1]
    x *= xmax / n / 2.
    y *= xmax / n / 2.
    return bl_from_x(x, y, antenna_diameter), [x, y]


def get_concentric_baselines_woffset(umin, umax, nq, ntheta, offset=0, log=True):
    # Setup grid
    dtheta = 2 * np.pi / ntheta

    if log:
        r = np.logspace(np.log10(umin), np.log10(umax), nq)
    else:
        r = np.linspace(umin, umax, nq)

    r[1:] += np.random.normal(scale=offset * (r[1:] - r[:-1]))

    theta = np.arange(ntheta) * dtheta

    X = np.outer(r, np.cos(theta))
    Y = np.outer(r, np.sin(theta))

    grid = np.array([X.flatten(), Y.flatten()])

    return grid


def get_random_symmetric_baselines(N, umin, umax, a, theta_dist='uniform'):
    """
    Generate a random symmetric set of baselines.

    Baselines will follow a power-law radial distribution, and uniform angular distribution,
    with possible step-function gaps to increase efficiency.
    """

    if theta_dist == 'uniform':
        theta = np.random.uniform(0, 2 * np.pi, size=N)
    elif type(theta_dist) == tuple:
        nspoke = np.random.randint(0, theta_dist[0], size=N)
        theta = np.random.uniform(0, theta_dist[1], size=N)
        theta += nspoke * 2 * np.pi / theta_dist[0]
        theta[theta > 2 * np.pi] -= 2 * np.pi
    elif type(theta_dist) == int:
        nspoke = np.random.randint(0, theta_dist, size=N)
        theta = nspoke * 2 * np.pi / theta_dist

    r = sample_pareto_minmax(a, umin, umax, size=N)

    X = r * np.cos(theta)
    Y = r * np.sin(theta)

    grid = np.array([X, Y])


def get_concentric_baselines(umin, umax, nq, ntheta, log=True):
    # Setup grid
    dtheta = 2 * np.pi / ntheta

    if log:
        r = np.logspace(np.log10(umin), np.log10(umax), nq)
    else:
        r = np.linspace(umin, umax, nq)

    theta = np.arange(ntheta) * dtheta

    X = np.outer(r, np.cos(theta))
    Y = np.outer(r, np.sin(theta))

    grid = np.array([X.flatten(), Y.flatten()])

    return grid
