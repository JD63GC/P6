import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# PARAMETERS
# =========================

# Geometry
rE = 6400e3
ra = 550e3
rS = rE + ra

# Monte Carlo
n_iter = 10000

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

# Transmit power (30 dBW → Watt)
Pt = 10**(30/10) / 1000   # = 1 W

# Antenna gain (20 dB → linear)
g = 10**(20/10)           # = 100

# Carrier frequency
f = 2e9
c = 3e8

# Path-loss constant (Friis)
K = (c / (4 * np.pi * f))**2

# Noise (kTB)
k = 1.38e-23
T = 290
B = 1e5   # 10 MHz

sigma2 = k * T * B

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
def simulate_coverage(lambda_, rS, rE, tau, n_iter, mode="SIR"):
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

            # nearest satellite
            idx0 = np.argmin(d_visible)
            d0 = d_visible[idx0]

            # Rayleigh fading
            H0 = np.random.exponential(1)

            # =====================
            # SNR
            # =====================
            if mode == "SNR":
                signal = Pt * g * K * H0 * d0**(-alpha)
                SNR = signal / sigma2

                if SNR > t:
                    count += 1

            # =====================
            # SIR
            # =====================
            elif mode == "SIR":
                interferers = np.delete(d_visible, idx0)

                if len(interferers) == 0:
                    count += 1
                    continue

                Hi = np.random.exponential(1, size=len(interferers))

                signal = Pt * g * K * H0 * d0**(-alpha)
                interference = np.sum(Pt * K * Hi * interferers**(-alpha))

                SIR = signal / interference

                if SIR > t:
                    count += 1

        results.append(count / n_iter)

    return results


# =========================
# RUN SIMULATIONS
# =========================
print("Running SNR simulation...")
snr_results = simulate_coverage(lambda_, rS, rE, tau, n_iter, mode="SNR")

print("Running SIR simulation...")
sir_results = simulate_coverage(lambda_, rS, rE, tau, n_iter, mode="SIR")


# =========================
# PLOTS
# =========================

# --- SNR ---
plt.figure()
plt.scatter(tau_dB, snr_results, color='red', s=20)
plt.xlabel(r"$\tau$ [dB]")
plt.ylabel(r"$P(\mathrm{SNR} > \tau)$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("PPP_SNR_physical.pdf")
plt.show()

# --- SIR ---
plt.figure()
plt.scatter(tau_dB, sir_results, color='red', s=20)
plt.xlabel(r"$\tau$ [dB]")
plt.ylabel(r"$P(\mathrm{SIR} > \tau)$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("PPP_SIR_physical.pdf")
plt.show()