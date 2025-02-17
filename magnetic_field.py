import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# =============================================================================
# Physical Parameters (Geometric Units: G = c = 1)
# =============================================================================
M = 1.0      # Mass
a = 0.8      # Spin (a = J/M)
Q = 0.3      # Charge

# Check black hole condition
if M**2 < Q**2 + a**2:
    raise ValueError("M² must be ≥ Q² + a² to avoid a naked singularity.")

# =============================================================================
# Spacetime Geometry
# =============================================================================
r_plus = M + np.sqrt(M**2 - a**2 - Q**2)  # Event horizon

# Create uniform Cartesian grid
x = np.linspace(-5*M, 5*M, 150)
y = np.linspace(-5*M, 5*M, 150)
X, Y = np.meshgrid(x, y)

# Convert to polar coordinates (r, θ)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# =============================================================================
# Magnetic Field Calculation (ZAMO Frame)
# =============================================================================
valid_mask = (R >= r_plus)  # Valid region (excludes inside the horizon)
Sigma = R**2 + (a**2) * (np.cos(Theta)**2)

B_r = np.zeros_like(R)
B_theta = np.zeros_like(R)
B_r[valid_mask] = (Q * a * np.cos(Theta[valid_mask]) / Sigma[valid_mask]**2) * (R[valid_mask]**2 - a**2 * np.cos(Theta[valid_mask])**2)
B_theta[valid_mask] = (Q * a * R[valid_mask] * np.sin(Theta[valid_mask]) / Sigma[valid_mask]**2) * (R[valid_mask]**2 - a**2 * np.cos(Theta[valid_mask])**2)

Bx = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
By = B_r * np.sin(Theta) + B_theta * np.cos(Theta)

B_magnitude = np.sqrt(Bx**2 + By**2)
B_magnitude[B_magnitude == 0] = 1e-10  # Avoid NaN

# =============================================================================
# Visualization
# =============================================================================
plt.figure(figsize=(12, 10))

# Streamlines
strm = plt.streamplot(X, Y, Bx, By, color=np.log10(B_magnitude), 
                     cmap='plasma', linewidth=1.5, density=2.5, 
                     arrowstyle='->', arrowsize=1.5,
                     norm=colors.Normalize(vmin=-3, vmax=np.log10(np.max(B_magnitude))))

# Event horizon and ergosphere
theta = np.linspace(0, 2*np.pi, 300)
x_horizon = r_plus * np.cos(theta)
y_horizon = r_plus * np.sin(theta)
r_ergo = M + np.sqrt(M**2 - (a**2) * (np.cos(theta)**2) - Q**2)
x_ergo = r_ergo * np.cos(theta)
y_ergo = r_ergo * np.sin(theta)

plt.plot(x_horizon, y_horizon, 'r--', lw=2.5, label='Event Horizon')
plt.plot(x_ergo, y_ergo, 'b--', lw=2.5, label='Ergosphere')

# Final adjustments
plt.title(f'Magnetic Field Topology: Kerr-Newman ($a={a}$, $Q={Q}$)', fontsize=14)
plt.xlabel('$x$ (M)', fontsize=12)
plt.ylabel('$y$ (M)', fontsize=12)
cbar = plt.colorbar(strm.lines, label='log$_{10}$(Field Magnitude)')
cbar.set_ticks(np.linspace(-3, np.log10(np.max(B_magnitude)), 5))
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(alpha=0.15, linestyle='--')
plt.axis('equal')
plt.tight_layout()
plt.show()