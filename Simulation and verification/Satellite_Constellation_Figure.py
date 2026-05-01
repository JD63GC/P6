import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
rE = 6400e3
ra = 550e3
rS = rE + ra

mu = 250
lambda_ = mu / (4 * np.pi * rS**2)

# =========================
# SAMPLE SATELLITES
# =========================
def sample_satellites(lambda_, rS):
    N = np.random.poisson(4 * np.pi * rS**2 * lambda_)
    
    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(-1, 1, N)
    phi = np.arccos(u)
    
    x = rS * np.sin(phi) * np.cos(theta)
    y = rS * np.sin(phi) * np.sin(theta)
    z = rS * np.cos(phi)
    
    return x, y, z

x, y, z = sample_satellites(lambda_, rS)

# =========================
# SPHERE FUNCTION
# =========================
def create_sphere(radius, resolution=50):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    
    return xs, ys, zs

# Create spheres
xE, yE, zE = create_sphere(rE)
xS, yS, zS = create_sphere(rS)

# =========================
# PLOT
# =========================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Earth (inner sphere)
ax.plot_surface(xE, yE, zE, color='lightblue', alpha=0.3, linewidth=0)

# Satellite sphere (wireframe)
ax.plot_wireframe(xS, yS, zS, color='gray', alpha=0.3)

# Satellites
ax.scatter(x, y, z, color='orange', s=20)

# Styling
ax.set_box_aspect([1,1,1])
ax.set_axis_off()

# Zoom correctly
ax.set_xlim([-rS, rS])
ax.set_ylim([-rS, rS])
ax.set_zlim([-rS, rS])

# Better view
ax.view_init(elev=20, azim=30)
plt.show()