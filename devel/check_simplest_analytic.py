from power_integral import power_single_baseline
import numpy as np
from make_power_spectra import generate_2d_ps


from spore.model.source_counts import PowerLawSourceCounts

import matplotlib.pyplot as plt

u = np.logspace(1,3, 10)
f = np.linspace(1, 16./15., 12)

tau = 100.0
sigma = 0.2

nreal = 500

u0 = np.vstack((u, np.zeros_like(u)))
power = [0]*nreal
for i in range(nreal):
    power[i], weights, omega = generate_2d_ps(
        u0, f, sigma, u, theta = [0], taper=np.exp(-tau ** 2 * (f - 1) ** 2),
        extent=50
    )

numerical_power = np.mean(np.array(power), axis=0)

# get a fiducial source-count model.
sc = PowerLawSourceCounts(1, 8e-1, 1, 0, alpha=4100.0, beta=1.59)

# Get analytic answer
analytic = np.zeros((len(u), len(omega)))
for iu in range(len(u)):
    analytic[iu] = power_single_baseline(u[iu], omega, tau, sigma, sc, nu0=1)

print(numerical_power.shape, analytic.shape)
#print(numerical_power, analytic)

cols = 'rgbmk'
for i in range(len(omega)):
    plt.plot(u, numerical_power[i,:,0], ls='-', color=cols[i])
    plt.plot(u, analytic[:, i], ls='--', color=cols[i])

plt.xscale('log')
plt.yscale('log')
plt.savefig("tmp_check.png")
