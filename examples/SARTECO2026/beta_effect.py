import numpy as np
import matplotlib.pyplot as plt

B = 6
gamma = 12
tau = np.arange(0, 9, 1)

beta_values = [1.0, 0.9, 0.8, 0.7]

plt.figure(figsize=(6,4))

colors = plt.cm.Blues(np.linspace(0.4, 1, len(beta_values)))

for beta, color in zip(beta_values, colors):
    mu = np.exp(-beta * (tau + np.log(1 + B / gamma)))
    plt.plot(
        tau, mu,
        marker='o',
        color=color,
        label=rf"$\beta={beta}$"
    )

plt.xlabel(r'$\tau_i$')
plt.ylabel(r'$\mu_i$')
plt.xticks(tau)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('beta_effect.pdf')
plt.close()

import numpy as np
import matplotlib.pyplot as plt

B = 6
beta = 0.8
tau = np.arange(0, 9, 1)

gamma_values = [6, 12, 18]

plt.figure(figsize=(6,4))

colors = plt.cm.Greens(np.linspace(0.4, 1, len(gamma_values)))

for gamma, color in zip(gamma_values, colors):
    mu = np.exp(-beta * (tau + np.log(1 + B / gamma)))
    plt.plot(
        tau, mu,
        marker='s',
        color=color,
        label=rf"$\gamma={gamma}$"
    )

plt.xlabel(r'$\tau_i$')
plt.ylabel(r'$\mu_i$')
plt.xticks(tau)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('gamma_effect.pdf')
plt.close()