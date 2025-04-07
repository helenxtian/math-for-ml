import numpy as np
import matplotlib.pyplot as plt

# n = number of points
# z = points where polynomial is evaluated
# p = array to store the values of the interpolated polynomials
n = 100
z = np.linspace(-10, 10, n)

d = 3 # degree
rng = np.random.default_rng() # random number generator
w = rng.uniform(-1, 1, d+1)
X = np.zeros((n, d+1))

powers = np.arange(d+1)
X = z.reshape(-1,1) ** powers
p = np.dot(X, w)

plt.plot(z, p, linewidth=2)
plt.xlabel('z')
plt.ylabel('y')
plt.title('Polynomial with coefficients w = %s' % w)
plt.show()