"""
Script used to get the 2D power spectrum for the fiducial "sparse" array to which everything is compared in Sec. 5
of the paper.
"""

from numerical import weighted_gridding_paper


nreal = 200

for i in range(nreal):
    weighted_gridding_paper(u0, f, sigma, tau, umin, umax, nu=1, ntheta=50, extent=50)
