from functools import partial
import numpy as np
from multiprocessing import Pool
import h5py
from .make_power_spectra import generate_2d_power, generate_2d_power_sparse, generate_3d_power


def seed_wgp(seed=1, **kwargs):
    np.random.seed(seed)
    return generate_2d_power(**kwargs)


def seed_wgp_sparse(seed=1, **kwargs):
    np.random.seed(seed)
    return generate_2d_power_sparse(**kwargs)

def imap_unordered(fnc, iterable):
    "Mock of multi-processing imap_unordered without using a pool"

def numerical_power_vec(*, fname, u0, umin, umax, nu, taper, sigma, f, realisations, nthreads, processes=1, restart=False,
                        extent=50, sky_moment=1, Smax=1):
    done = np.zeros(realisations)

    with h5py.File(fname, 'w' if restart else 'a') as fl:
        if "power" not in fl:
            fl.attrs['sigma'] = sigma
            fl.attrs['nu'] = nu
            fl.attrs['realisations'] = realisations
            fl.attrs['extent'] = extent
            fl.attrs['moment'] = sky_moment
            fl.attrs['Smax'] = Smax

            fl.create_dataset("power", data=np.zeros((realisations, nu, int(len(f)/2))))
            fl.create_dataset("weights", data=np.zeros(nu))
            fl.create_dataset("u", data=np.zeros(nu))
            fl.create_dataset("omega", data=np.zeros(int(len(f)/2)))
            fl.create_dataset("u0", data=u0)
            fl.create_dataset('f', data=f)
            fl.create_dataset("done", data=done)

        else:
            assert fl.attrs['realisations'] == realisations
            assert fl.attrs['sigma'] == sigma
            assert fl.attrs['nu'] == nu
            assert fl.attrs['moment'] == sky_moment
            assert fl.attrs['Smax'] == Smax

            done[...] = fl['done'][...]

    if np.sum(done)/done.size == 1:
        print("This computation has already been performed. Exiting. Run with --restart to restart.")
    else:
        np.random.seed(1234)  # Set this seed so the same thing happens every time.
        seeds = np.random.randint(0, 20000 * realisations, size=int(realisations))

        j = np.sum(done)

        # Only use as many processes as there are iterations, at max.
        processes = min(processes, realisations - j)

        fnc = partial(seed_wgp, u0=u0, f=f, sigma=sigma, taper=taper, umin=umin, umax=umax, nu=nu, ntheta=100,
                      extent=extent, nthreads=nthreads, moment=sky_moment, Smax=Smax)

        def save_stuff(p, omega, weights):
            with h5py.File(fname, 'a') as fl:
                fl['power'][j, :, :] = p[...].T

                if not j:
                    fl['weights'][...] = weights[...]
                    fl['u'][...] = np.logspace(np.log10(umin), np.log10(umax), nu)
                    fl['omega'][...] = omega[...]

                fl['done'][j] = 1

            print("Done %s of %s iterations." % (j+1, realisations))

        if processes > 1:
            pl = Pool(processes)

            for p, omega, weights in pl.imap_unordered(fnc, seeds[int(np.sum(done)):]):
                save_stuff(p, omega, weights)
                j += 1

            pl.close()
            pl.join()
        else:
            for p, omega, weights in map(fnc, seeds[int(np.sum(done)):]):
                save_stuff(p, omega, weights)
                j += 1

def numerical_sparse_power_vec(*, fname, umin, umax, nu, taper, sigma, f, realisations, nthreads, processes=1,
                               restart=False, extent=50, sky_moment=1, Smax=1):
    done = np.zeros(realisations)

    with h5py.File(fname, 'w' if restart else 'a') as fl:
        if "power" not in fl:
            fl.attrs['sigma'] = sigma
            fl.attrs['nu'] = nu
            fl.attrs['realisations'] = realisations
            fl.attrs['moment'] = sky_moment
            fl.attrs['Smax'] = Smax

            fl.create_dataset("power", data=np.zeros((realisations, nu, int(len(f)/2))))
            fl.create_dataset("u", data=np.zeros(nu))
            fl.create_dataset("omega", data=np.zeros(int(len(f)/2)))
            fl.create_dataset("done", data=done)

        else:
            assert fl.attrs['realisations'] == realisations
            assert fl.attrs['sigma'] == sigma
            assert fl.attrs['nu'] == nu
            assert fl.attrs['Smax'] == Smax
            assert fl.attrs['moment'] == sky_moment

            done[...] = fl['done'][...]

    if np.sum(done)/done.size == 1:
        print("This computation has already been performed. Exiting. Run with --restart to restart.")
    else:
        np.random.seed(4321)  # Set this seed so the same thing happens every time.
        seeds = np.random.randint(0, 20000 * realisations, size=int(realisations))

        pl = Pool(processes)
        j = np.sum(done)

        fnc = partial(seed_wgp_sparse, f=f, sigma=sigma, taper=taper, umin=umin, umax=umax, nu=nu, ntheta=100,
                      extent=extent, nthreads=nthreads, moment=sky_moment, Smax=Smax)

        for p, omega in pl.imap_unordered(fnc, seeds[int(np.sum(done)):]):

            with h5py.File(fname, 'a') as fl:
                fl['power'][j, :, :] = np.transpose(p[...], (0,2,1))

                if not j:
                    fl['u'][...] = np.logspace(np.log10(umin), np.log10(umax), nu)
                    fl['omega'][...] = omega[...]

                fl['done'][j] = 1

            j += 1
            print("Done %s of %s iterations." % (j, realisations))

        pl.close()
        pl.join()