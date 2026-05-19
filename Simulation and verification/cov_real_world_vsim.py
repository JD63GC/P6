# ============================================================
# STARLINK + PPP COVERAGE PROBABILITY
# MED REPRODUCERBARE RANDOM SNAPSHOTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from skyfield.api import Topos, load
import requests
import os
from datetime import datetime, timedelta
seed=41
np.random.seed(seed)
rng = np.random.default_rng(seed)

# ============================================================
# PARAMETERS
# ============================================================

tuning_factor = 0.6
alpha = 2

G_serving = 1.0
G_interferer = 0.1

aalborg = Topos(latitude_degrees=57.0488,
                longitude_degrees=9.9217)

elevation_mask_deg = 0
reuse_range = 20

num_realizations = 15000
NUM_SNAPSHOTS = 50


gamma_db = np.arange(-10, 21, 2)
gamma_lin = 10**(gamma_db / 10)

R_E = 6371e3
h = 550e3
R_S = R_E + h

# ============================================================
# LOAD TLE (deterministisk fil anbefales også)
# ============================================================

tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
satellites = load.tle_file(tle_url)

ts = load.timescale()

# ============================================================
# FIXED RANDOM SNAPSHOTS (JANUARY 2026)
# ============================================================

START_DATE = datetime(2026, 1, 1)
END_DATE   = datetime(2026, 1, 31)

time_span = (END_DATE - START_DATE).total_seconds()

snapshot_times = []

for _ in range(NUM_SNAPSHOTS):
    dt = START_DATE + timedelta(
        seconds=float(rng.uniform(0, time_span))
    )
    snapshot_times.append(dt)

# ============================================================
# CHANNEL
# ============================================================

def fading(N):
    return rng.exponential(1, N)

# ============================================================
# VISIBILITY FUNCTION (depends on global t)
# ============================================================

def get_visible_distances(t):

    visible = []

    for sat in satellites:

        difference = sat - aalborg
        topocentric = difference.at(t)

        alt, az, distance = topocentric.altaz()

        if alt.degrees > elevation_mask_deg:
            visible.append(distance.m)

    return np.array(visible)

# ============================================================
# STARLINK SIR
# ============================================================

def compute_starlink_sir(visible_distances):

    if len(visible_distances) == 0:
        return 0

    distances = np.copy(visible_distances)

    serving_idx = np.argmin(distances)
    serving_distance = distances[serving_idx]

    H_signal = rng.exponential(1)
    signal = G_serving * H_signal * serving_distance**(-alpha)

    interferer_indices = []

    for i in range(len(distances)):

        if i == serving_idx:
            continue

        if rng.integers(1, reuse_range + 1) == 1:
            interferer_indices.append(i)


    if len(interferer_indices) == 0:
        return 1e9

    interferer_distances = distances[interferer_indices]
    H_interference = fading(len(interferer_distances))

    interference = np.sum(
        G_interferer * H_interference * interferer_distances**(-alpha)
    )

    return signal / interference

# ============================================================
# PPP GEOMETRY
# ============================================================

eps = np.radians(elevation_mask_deg)

theta_max = np.arccos(
    (R_E / R_S) * np.cos(eps)**2
    + np.sin(eps) * np.sqrt(
        1 - (R_E / R_S)**2 * np.cos(eps)**2
    )
)

visible_area = 2 * np.pi * R_S**2 * (1 - np.cos(theta_max))

# ============================================================
# STARLINK SIMULATION
# ============================================================

print("Running snapshot-averaged simulation...")

all_starlink_cov = []

for idx, random_time in enumerate(snapshot_times):

    t = ts.utc(
        random_time.year,
        random_time.month,
        random_time.day,
        random_time.hour,
        random_time.minute,
        random_time.second
    )

    visible_distances = get_visible_distances(t)

    # Monte Carlo per snapshot
    sir_samples = []

    for _ in range(num_realizations):
        sir_samples.append(
            compute_starlink_sir(visible_distances)
        )

    sir_samples = np.array(sir_samples)

    cov = [
        np.mean(sir_samples > g)
        for g in gamma_lin
    ]

    all_starlink_cov.append(cov)

    if idx % 10 == 0:
        print(f"Snapshot {idx+1}/{NUM_SNAPSHOTS}")

all_starlink_cov = np.array(all_starlink_cov)

starlink_cov = np.mean(all_starlink_cov, axis=0)
#---------------------------------------------------
# PPP MODEL
#---------------------------------------------------
mu_target = len(visible_distances)
lambda_ppp = mu_target / visible_area

def generate_ppp_distances(lambda_density):

    N = poisson.rvs(lambda_density * visible_area, random_state=rng)

    if N == 0:
        return np.array([])

    u = rng.uniform(np.cos(theta_max), 1, N)
    phi = np.arccos(u)

    return np.sqrt(
        R_E**2 + R_S**2 - 2 * R_E * R_S * np.cos(phi)
    )

def compute_ppp_sir(lambda_density):

    distances = generate_ppp_distances(lambda_density)

    if len(distances) == 0:
        return 0

    serving_idx = np.argmin(distances)
    serving_distance = distances[serving_idx]

    H_signal = rng.exponential(1)
    signal = G_serving * H_signal * serving_distance**(-alpha)

    interferer_indices = []

    for i in range(len(distances)):

        if i == serving_idx:
            continue

        if rng.integers(1, reuse_range + 1) == 1:
            interferer_indices.append(i)

    if len(interferer_indices) == 0:
        return 1e9

    interferer_distances = distances[interferer_indices]
    H_interference = fading(len(interferer_distances))

    interference = np.sum(
        G_interferer * H_interference * interferer_distances**(-alpha)
    )

    return signal / interference

def simulate_ppp_coverage(lambda_density):

    sir_samples = [
        compute_ppp_sir(lambda_density)
        for _ in range(num_realizations)
    ]

    sir_samples = np.array(sir_samples)

    return np.array([
        np.mean(sir_samples > g)
        for g in gamma_lin
    ])

# ============================================================
# RUN
# ============================================================

ppp_cov = simulate_ppp_coverage(lambda_ppp)
lambda_tuned = tuning_factor * lambda_ppp
ppp_tuned_cov = simulate_ppp_coverage(lambda_tuned)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(gamma_db, ppp_cov, 'bs--', label='PPP')
plt.plot(gamma_db, starlink_cov, 'k*--', label='Starlink')

plt.xlabel(r'$\gamma$ (dB)')
plt.ylabel("Coverage Probability")
plt.grid(True)
plt.legend()
plt.ylim([0, 1.05])

plt.title(
    rf"λ={lambda_ppp:.2e}, μ={mu_target:.1f}, mask={elevation_mask_deg}°"
)

plt.tight_layout()
plt.show()

#Figure 2: Tuning lambda for better fit

plt.figure(figsize=(7, 5))

plt.plot(gamma_db, ppp_tuned_cov, 'bs--', color='r', label='PPP')
plt.plot(gamma_db, starlink_cov, 'k*--', label='Starlink')

plt.xlabel(r'$\gamma$ (dB)')
plt.ylabel("Coverage Probability")
plt.grid(True)
plt.legend()
plt.ylim([0, 1.05])

plt.title(
    rf"λ tuned with tuning_factor={tuning_factor:.2f}"
)

plt.tight_layout()
plt.show()