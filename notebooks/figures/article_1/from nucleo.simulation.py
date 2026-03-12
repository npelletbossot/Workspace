from nucleo.simulation.probabilities import proba_gamma
import matplotlib.pyplot as plt
import numpy as np


L = np.arange(0.0, 1000 + 0.1, 1)

# p = proba_gamma(mu = 160, theta = 90, L = L)
# plt.figure(figsize=(8,6))
# plt.plot(L, p)
# plt.show()

p = proba_gamma(mu = 160, theta = 500, L = L)
plt.figure(figsize=(8,6))
plt.plot(L, p)
plt.show()
