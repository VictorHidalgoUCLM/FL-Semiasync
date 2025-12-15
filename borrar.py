import numpy as np
import matplotlib.pyplot as plt

# Parámetros
beta = 0.75
total_bias = 7.0
k = 7.0

# Rango de counter_val
counter_vals = np.linspace(0, 20, 400)

# Cálculo de mu_i
mu_vals = np.exp(-beta * (counter_vals + np.log1p(total_bias / k)))

# Plot
plt.figure(figsize=(7, 4))
plt.plot(counter_vals, mu_vals, linewidth=2)
plt.title(r"Factor de penalización $\mu_i$")
plt.xlabel("counter_val")
plt.ylabel(r"$\mu_i$")
plt.grid(True)
plt.tight_layout()
plt.savefig("mu_i_plot.png")
