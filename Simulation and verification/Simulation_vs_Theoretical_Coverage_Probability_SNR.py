import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

# =========================
# PARAMETERS
# =========================

# Geometry
rE = 6400e3
ra = 550e3
rS = rE + ra

# Monte Carlo
n_iter = 1000

# PPP
mu = 190
lambda_ = mu / (4 * np.pi * rS**2)

# Thresholds
tau_dB = np.linspace(-10, 20, 30)
tau = 10**(tau_dB / 10)

# Path-loss exponent
alpha = 2

# =========================
# PHYSICAL PARAMETERS
# =========================

Pt = 10**(30/10) / 1000   # 1 W
g = 10**(20/10)           # 100

f = 2e9
c = 3e8
K = (c / (4 * np.pi * f))**2

k = 1.38e-23
T = 290
B = 1e5

sigma2 = k * T * B

# =========================
# GEOMETRY HELPERS
# =========================

R_min = rS - rE
R_max = np.sqrt(rS**2 - rE**2)

area_cap = 2 * np.pi * rS * (rS - rE)

# =========================
# SAMPLE PPP ON SPHERE
# =========================
def sample_satellites(lambda_, rS):
    N = np.random.poisson(4 * np.pi * rS**2 * lambda_)

    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(-1, 1, N)
    phi = np.arccos(u)

    x = rS * np.sin(phi) * np.cos(theta)
    y = rS * np.sin(phi) * np.sin(theta)
    z = rS * np.cos(phi)

    return np.vstack((x, y, z)).T


# =========================
# COVERAGE SIMULATION
# =========================
def simulate_coverage(lambda_, rS, rE, tau, n_iter, mode="SNR"):
    results = []

    user = np.array([0, 0, rE])
    R_max = np.sqrt(rS**2 - rE**2)

    for t in tqdm(tau):
        count = 0

        for _ in range(n_iter):
            sats = sample_satellites(lambda_, rS)

            if len(sats) == 0:
                continue

            dists = np.linalg.norm(sats - user, axis=1)
            visible = dists <= R_max

            if np.sum(visible) == 0:
                continue

            d_visible = dists[visible]

            idx0 = np.argmin(d_visible)
            d0 = d_visible[idx0]

            H0 = np.random.exponential(1)

            if mode == "SNR":
                signal = Pt * g * K * H0 * d0**(-alpha)
                SNR = signal / sigma2

                if SNR > t:
                    count += 1

        results.append(count / n_iter)

    return results


# =========================
# THEORETICAL SNR
# =========================

# Distance PDF (approximation).         !!!DER HER SKAL MÅSKE ÆNDRES FORDI DET ER EN APPROKSIMATION!!!
def f_R(r):
    C = 2 * np.pi * lambda_ * rS / rE
    return C * r * np.exp(-lambda_ * np.pi * rS / rE * (r**2 - R_min**2))


def snr_coverage_theory(tau):

    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):
        return np.exp(-tau * sigma2 * r**alpha / (Pt * g * K)) * f_R(r)

    integral, _ = quad(integrand, R_min, R_max, limit=100)

    return P_visible * integral


# =========================
# RUN SIMULATION
# =========================

print("Running SNR simulation...")
snr_results = simulate_coverage(lambda_, rS, rE, tau, n_iter, mode="SNR")

print("Computing theoretical curve...")
snr_theory = np.array([snr_coverage_theory(t) for t in tau])


# =========================
# PLOT
# =========================

plt.figure()
plt.scatter(tau_dB, snr_results, color='red', s=20, label='Simulation')
plt.plot(tau_dB, snr_theory, color='blue', linewidth=2, label='Theory')
plt.xlabel(r"$\tau$ [dB]")
plt.ylabel(r"$P(\mathrm{SNR} > \tau)$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig("PPP_SNR_comparison.pdf")
plt.show()