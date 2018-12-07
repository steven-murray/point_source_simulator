# Init ctypes types
import ctypes as ct
import os
import numpy as np
import glob

cdll = ct.CDLL(glob.glob(os.path.join(os.path.dirname(__file__), "get_visibilities.*.so"))[0])

cdll.getvis.argtypes = [
    ct.c_int, ct.c_int, ct.c_int,  np.ctypeslib.ndpointer(np.float64),
    np.ctypeslib.ndpointer(np.float64),  np.ctypeslib.ndpointer(np.float64),   np.ctypeslib.ndpointer(np.float64),
    np.ctypeslib.ndpointer(np.float64),  np.ctypeslib.ndpointer(np.float64), ct.c_int, np.ctypeslib.ndpointer(np.complex128)
]

cdll.get_baselines.argtypes = [
    ct.c_uint, np.ctypeslib.ndpointer(np.float64),  np.ctypeslib.ndpointer(np.float64),   ct.c_double,
    np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.float64)
]

cdll.get_bad_antennas.argtypes = [
    ct.c_uint, np.ctypeslib.ndpointer(np.float64),  np.ctypeslib.ndpointer(np.float64),   ct.c_double, ct.c_int,
    np.ctypeslib.ndpointer(np.int32)
]

cdll.get_baselines.restype = ct.c_int64


def get_visibilities(f, u0, source_flux, source_pos, nthreads=1):
    """
    Generate visibilities from a list of point sources and their apparent flux densities.

    Parameters
    ----------
    f: array
        Frequencies of observation, normalised by reference frequency.

    u0 : array, shape == (nbl,2)
        The x,y co-ordinates of the baselines, in units of wavelengths, at reference frequency (f=1).

    source_flux : array
        The *apparent* flux of each point-source.

    source_pos : array, shape == (nsource, 2)
        Positions, in sin-projected units, of the sources on the sky.

    Returns
    -------
    vis : complex array, shape = (nf, nbl)
        The complex visibilities.
    """

    vis = np.zeros((len(f),len(u0)), dtype=np.complex128).flatten()

    cdll.getvis(len(f), len(u0), len(source_flux), f, 2 * u0[:, 0] * np.pi, 2 * u0[:, 1] * np.pi,
                source_flux, source_pos[:,0], source_pos[:,1], nthreads, vis)

    return vis.reshape((len(f), len(u0)))


def get_baselines(x, y, antenna_diameter):
    nant = len(x)
    bl = np.zeros((2, nant*(nant-1)))

    res = cdll.get_baselines(nant, x, y, antenna_diameter, bl[0], bl[1])

    if res <= 0:
        return res
    else:
        bl[:, res:] = -bl[:, :res]
        return bl


def get_good_antennas(x, y, antenna_diameter, start):
    bad_antennas = np.zeros(len(x), dtype=np.int32)

    cdll.get_bad_antennas(len(x), x, y, antenna_diameter, start, bad_antennas)

    bad_antennas = bad_antennas.astype(dtype='bool')

    return x[~bad_antennas], y[~bad_antennas]
