# MWE for a single orbit-averaged evolution from a given GW frequency

import numpy as np
import precession as precession2
import surfinBH


f_gw = 20e-3 # in Hz

# Some random binary parameters at a reference GW frequency
m1 = 30
m2 = 25
chi1 = 0.8
chi2 = 0.8
theta1 = np.pi / 3
theta2 = np.pi / 4
phi1 = 0
phi2 = 3 * np.pi / 5

q = m2 / m1 # < 1
deltaphi = phi2 - phi1

# GW frequency is frame-dependent, so this total mass has to match
# If f_gw is detector-frame, M is redshifted total mass
# If f_gw is source-frame, M is source-frame total mass
M = m1 + m2 # solar mass units

# Convert GW frequency to (dimensionless) PN orbital separation r/M
r0 = precession2.gwfrequency_to_pnseparation(
    theta1, theta2, deltaphi, f_gw, q, chi1, chi2, M,
    )

# Output separations of spin evolution
r_vals = np.linspace(np.squeeze(r0), 1, 1000)

# Precession-averaged 'precession' or orbit-averaged 'orbit'
# If you do precession-averaged, the spin resampling is done for you by the code
which = 'orbit'

# Returns a dictionary of outputs
result = precession2.inspiral(
    which=which,
    theta1=theta1,
    theta2=theta2,
    deltaphi=deltaphi,
    r=r_vals,
    q=q,
    chi1=chi1,
    chi2=chi2,
    #requested_outputs=[], # list of parameters you want to output
    )

print(result.keys())

# Outputs are shaped as (n_binaries, n_r_vals)
theta1_vals = np.squeeze(result['theta1'])
theta2_vals = np.squeeze(result['theta2'])
deltaphi_vals = np.squeeze(result['deltaphi'])

# Conversion back to GW frequency depends on separation and spin evolutions
f_vals = precession2.gwfrequency_to_pnseparation(
    theta1_vals, theta2_vals, deltaphi_vals, r_vals, q, chi1, chi2, M,
    )

# Orbital frequency you want the spins at
omega0_choice = 0.03 # just an example for the remnant surrogate
omega0_vals = 4.93e-6 * np.pi * M * f_vals

# You could interpolate the spin evolutions to get them at omega0
# Here I just take the closest
idx = np.argmin(np.abs(omega0_vals - omega0_choice))
omega0 = omega0_vals[idx]
theta10 = theta1_vals[idx]
theta20 = theta2_vals[idx]
deltaphi0 = deltaphi_vals[idx]

# Convert spin angles + orbital phase to spin vectors in required frame
def angles_to_vectors(
    orbphi, theta1, theta2, deltaphi, q, chi1, chi2
    ):
    
    # Primary spin
    chiAx = chi1 * np.sin(theta1) * np.cos(orbphi)
    chiAy = chi1 * np.sin(theta1) * np.sin(orbphi)
    chiAz = chi1 * np.cos(theta1)
    # Secondary spin
    chiBx = chi2 * np.sin(theta2) * np.cos(orbphi + deltaphi)
    chiBy = chi2 * np.sin(theta2) * np.sin(orbphi + deltaphi)
    chiBz = chi2 * np.cos(theta2)
    
    return [chiAx, chiAy, chiAz], [chiBx, chiBy, chiBz]

# Resample orbital phase since it was averaged in the spin evolution
# For each posterior sample you need to evaluate the surrogate for many orbphi
orbphi = np.random.random() * 2 * np.pi

# Spin vectors in frame required by remnant surrogate model
chiA, chiB = angles_to_vectors(
    orbphi, theta10, theta20, deltaphi0, q, chi1, chi2,
    )

# Load remnant surrogate
fitname = 'NRSur7dq4Remnant'
fit = surfinBH.LoadFits(fitname)

# Remnant properties and their fit errors
mf, chif, vf, mf_err, chif_err, vf_err = fit.all(
    1/q, chiA, chiB, omega0=omega0, omega_switch_IG=omega0,
    )

