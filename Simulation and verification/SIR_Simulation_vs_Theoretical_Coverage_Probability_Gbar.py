import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

np.random.seed(42)

# =========================================================
# PARAMETERS
# =========================================================

# Earth / satellite geometry
rE = 6400e3                 # Earth radius [m]
ra = 550e3                  # satellite altitude [m]
rS = rE + ra                # orbital sphere radius

# Monte Carlo iterations
n_iter = 10

# PPP density
mu = 190                    # average number of satellites on sphere
lambda_ = mu / (4 * np.pi * rS**2)

# SIR thresholds
tau_dB = np.linspace(-10, 20, 30)
tau = 10**(tau_dB / 10)

# Path-loss exponent
alpha = 2

# Rayleigh fading
m = 1

# =========================================================
# ANTENNA GAIN MODEL (PAPER)
# =========================================================

# Interferers are 10 dB weaker than desired signal
# Paper uses G_bar = 0.1
G_bar = 0.1

# =========================================================
# GEOMETRY
# =========================================================

R_min = rS - rE
R_max = np.sqrt(rS**2 - rE**2)

# Area of visible spherical cap
area_cap = 2 * np.pi * rS * (rS - rE)

# =========================================================
# SAMPLE PPP ON SPHERE
# =========================================================

def sample_satellites(lambda_, rS):

    N = np.random.poisson(4 * np.pi * rS**2 * lambda_)

    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(-1, 1, N)
    phi = np.arccos(u)

    x = rS * np.sin(phi) * np.cos(theta)
    y = rS * np.sin(phi) * np.sin(theta)
    z = rS * np.cos(phi)

    return np.vstack((x, y, z)).T


# =========================================================
# CONDITIONAL DISTANCE PDF
# (Lemma 2 in paper)
# =========================================================

def nu_lambda_RS(lambda_, rS, rE):

    numerator = (
        2 * np.pi * lambda_ * rS / rE
        * np.exp(lambda_ * np.pi * rS / rE * (rS**2 - rE**2))
    )

    denominator = (
        np.exp(2 * lambda_ * np.pi * rS * (rS - rE)) - 1
    )

    return numerator / denominator


def f_R_conditional(r, lambda_, rS, rE):

    if r < R_min or r > R_max:
        return 0

    nu = nu_lambda_RS(lambda_, rS, rE)

    return nu * r * np.exp(
        -lambda_ * np.pi * rS / rE * r**2
    )


# =========================================================
# MONTE CARLO SIR SIMULATION
# =========================================================

def simulate_SIR(lambda_, rS, rE, tau, n_iter):

    results = []

    user = np.array([0, 0, rE])

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

            # =========================================
            # Serving satellite = nearest visible
            # =========================================

            idx0 = np.argmin(d_visible)

            d0 = d_visible[idx0]

            # Desired fading
            H0 = np.random.exponential(1)

            signal = H0 * d0**(-alpha)

            # =========================================
            # Interference
            # =========================================

            interferers = np.delete(d_visible, idx0)

            if len(interferers) == 0:
                count += 1
                continue

            Hi = np.random.exponential(1, size=len(interferers))

            interference = np.sum(
                G_bar * Hi * interferers**(-alpha)
            )

            SIR = signal / interference

            if SIR > t:
                count += 1

        results.append(count / n_iter)

    return np.array(results)


# =========================================================
# LAPLACE TRANSFORM OF INTERFERENCE
# (Lemma 3 in paper)
# =========================================================

def laplace_interference(s, r):

    factor = lambda_ * np.pi * rS / rE

    lower = (s * G_bar)**(-2/alpha) * r**2
    upper = (s * G_bar)**(-2/alpha) * R_max**2

    def integrand(u):

        return 1 - 1 / (1 + u**(-alpha/2))

    integral, _ = quad(integrand, lower, upper, limit=200)

    return np.exp(
        -factor * (s * G_bar)**(2/alpha) * integral
    )


# =========================================================
# THEORETICAL SIR COVERAGE
# (Corollary 1, m = 1)
# =========================================================

def sir_coverage_theory(tau):

    # Probability at least one visible satellite exists
    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):

        s = tau * r**alpha

        return (
            laplace_interference(s, r)
            * f_R_conditional(r, lambda_, rS, rE)
        )

    integral, _ = quad(
        integrand,
        R_min,
        R_max,
        limit=200
    )

    return P_visible * integral


# =========================================================
# RUN SIMULATION
# =========================================================

print("Running Monte Carlo simulation...")

sir_sim = simulate_SIR(
    lambda_,
    rS,
    rE,
    tau,
    n_iter
)

print("Computing theory...")

sir_theory = np.array([
    sir_coverage_theory(t)
    for t in tau
])

# =========================================================
# ERROR
# =========================================================

errors = np.abs(sir_sim - sir_theory)

max_error = np.max(errors)

print("\nMaximum absolute error:", max_error)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(7,5))

plt.scatter(
    tau_dB,
    sir_sim,
    color='red',
    s=25,
    label='Simulation'
)

plt.plot(
    tau_dB,
    sir_theory,
    color='blue',
    linewidth=2,
    label='Theory'
)

plt.xlabel(r"$\tau$ [dB]", fontsize=12)

plt.ylabel(r"$P(\mathrm{SIR} > \tau)$", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)

plt.legend()

plt.tight_layout()

plt.savefig("PPP_SIR_paper_match.pdf")

plt.show()