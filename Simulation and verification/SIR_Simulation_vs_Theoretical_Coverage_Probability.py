import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

np.random.seed(42)

# =========================
# PARAMETERS
# =========================

rE = 6400e3
ra = 550e3
rS = rE + ra

n_iter = 10000

mu = 190
lambda_ = mu / (4 * np.pi * rS**2)

tau_dB = np.linspace(-10, 20, 30)
tau = 10**(tau_dB / 10)

alpha = 2

# =========================
# GEOMETRY
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
# CONDITIONAL DISTANCE PDF
# =========================

def nu_lambda_RS(lambda_, rS, rE):
    numerator = (2 * np.pi * lambda_ * rS / rE *
                 np.exp(lambda_ * np.pi * rS / rE * (rS**2 - rE**2)))

    denominator = np.exp(2 * lambda_ * np.pi * rS * (rS - rE)) - 1

    return numerator / denominator


def f_R_conditional(r, lambda_, rS, rE, R_min, R_max):

    if r < R_min or r > R_max:
        return 0

    nu = nu_lambda_RS(lambda_, rS, rE)

    return nu * r * np.exp(-lambda_ * np.pi * rS / rE * r**2)


# =========================
# MONTE CARLO SIR
# =========================

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

            # Serving satellite
            idx0 = np.argmin(d_visible)
            d0 = d_visible[idx0]

            H0 = np.random.exponential(1)
            signal = H0 * d0**(-alpha)

            # Interference
            interference = 0
            for i in range(len(d_visible)):
                if i == idx0:
                    continue

                Hi = np.random.exponential(1)
                interference += Hi * d_visible[i]**(-alpha)

            if interference == 0:
                count += 1
                continue

            SIR = signal / interference

            if SIR > t:
                count += 1

        results.append(count / n_iter)

    return results


# =========================
# LAPLACE TRANSFORM
# =========================

def laplace_interference(s, r):

    factor = lambda_ * np.pi * rS / rE

    lower = (s)**(-2/alpha) * r**2
    upper = (s)**(-2/alpha) * R_max**2

    def integrand(u):
        return 1 - 1/(1 + u**(-alpha/2))

    integral, _ = quad(integrand, lower, upper, limit=100)

    return np.exp(-factor * s**(2/alpha) * integral)


# =========================
# THEORETICAL SIR
# =========================

def sir_coverage_theory(tau):

    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):
        s = tau * r**alpha

        return laplace_interference(s, r) * \
               f_R_conditional(r, lambda_, rS, rE, R_min, R_max)

    integral, _ = quad(integrand, R_min, R_max, limit=100)

    return P_visible * integral


# =========================
# RUN
# =========================

print("Running SIR simulation...")
sir_sim = simulate_SIR(lambda_, rS, rE, tau, n_iter)

print("Computing theory...")
sir_theory = np.array([sir_coverage_theory(t) for t in tau])


# =========================
# ERROR
# =========================

sir_sim = np.array(sir_sim)
errors = np.abs(sir_sim - sir_theory)

max_error = np.max(errors)
idx = np.argmax(errors)

print("Maximum absolute error (SIR):", max_error)


# =========================
# PLOT
# =========================

plt.figure()
plt.scatter(tau_dB, sir_sim, color='red', s=20, label='Simulation')
plt.plot(tau_dB, sir_theory, color='blue', linewidth=2, label='Theory')

plt.xlabel(r"$\tau$ [dB]")
plt.ylabel(r"$P(\mathrm{SIR} > \tau)$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig("PPP_SIR_comparison.pdf")
plt.show()