import numpy as np
import matplotlib.pyplot as plt

# Black hole parameters (geometric units: G = c = 1)
M = 1.0      # Mass
a = 0.9      # Spin parameter (a = J/M)
Q = 0.3      # Charge

# Outer horizon radius
r_plus = M + np.sqrt(M**2 - a**2 - Q**2)

# Create polar grid in Boyer-Lindquist coordinates
theta = np.linspace(0, 2*np.pi, 100)      # Azimuthal angle [0, 2π]
r = np.linspace(r_plus + 0.1, 5*M, 50)    # Radial distance [r_plus, 5M]
R, Theta = np.meshgrid(r, theta)           # Grid in (r, θ)

# Convert to Cartesian coordinates for plotting
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Compute Σ = r² + a² cos²θ
Sigma = R**2 + (a**2) * (np.cos(Theta)**2)

# Calculate magnetic field components B_hat{r} and B_hat{θ} (ZAMO frame)
B_r = (Q * a * np.cos(Theta) / Sigma**2) * (R**2 - a**2 * np.cos(Theta)**2)
B_theta = (Q * a * R * np.sin(Theta) / Sigma**2) * (R**2 - a**2 * np.cos(Theta)**2)

# Convert to Cartesian components (Bx, By)
Bx = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
By = B_r * np.sin(Theta) + B_theta * np.cos(Theta)

# Normalize vectors for visualization
norm = np.sqrt(Bx**2 + By**2)
Bx_normalized = Bx / norm
By_normalized = By / norm

# Plotting
plt.figure(figsize=(10, 8))

# Magnetic field lines (quiver plot)
plt.quiver(X, Y, Bx_normalized, By_normalized, norm, 
           cmap='viridis', scale=30, width=0.005, 
           clim=(0, np.max(norm)), 
           label='Magnetic Field')

# Event horizon (r = r_plus)
theta_horizon = np.linspace(0, 2*np.pi, 100)
x_horizon = r_plus * np.cos(theta_horizon)
y_horizon = r_plus * np.sin(theta_horizon)
plt.plot(x_horizon, y_horizon, 'r--', linewidth=2, label='Event Horizon')

# Ergosphere boundary (r = M + sqrt{M² - a² cos²θ - Q²})
r_ergo = M + np.sqrt(M**2 - (a**2) * (np.cos(theta_horizon)**2) - Q**2)
x_ergo = r_ergo * np.cos(theta_horizon)
y_ergo = r_ergo * np.sin(theta_horizon)
plt.plot(x_ergo, y_ergo, 'b--', linewidth=2, label='Ergosphere')

# Formatting
plt.title(f'Magnetic Field Structure: Kerr-Newman Black Hole ($a={a}$, $Q={Q}$)', fontsize=14)
plt.xlabel('$x$ (M)', fontsize=12)
plt.ylabel('$y$ (M)', fontsize=12)
plt.colorbar(label='Field Strength (arb. units)')
plt.legend(loc='upper right')
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(alpha=0.2)
plt.show()