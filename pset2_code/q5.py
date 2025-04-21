import numpy as np
import matplotlib.pyplot as plt

# File available on Canvas
data = np.load('polydata_a24.npz') 
x1 = np.ravel(data['x1'])
x2 = np.ravel(data['x2'])
y = data['y']

N = x1.size
p = np.zeros((3,N))

for d in [1 ,2 ,3]:
    # Generate the X matrix for this d
    # Note that here d is the degree of the polynomial, not the dimension of a vector
    # Find the least-squares weight matrix w_d
    # Evaluate the best-fit polynomial at each point (x1,x2) # and store the result in the corresponding column of p
    # Report the relative error of the polynomial fit

    X = np.zeros((N, 2 * d + 1))
    for i in range(N):
        X[i, 0] = 1.0
        for j in range(1, d + 1):
            X[i, j] = x1[i] ** j
            X[i, d + j] = x2[i] ** j

    # (X^T X)^-1 X^T y
    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)
    w_d = np.dot(np.linalg.inv(XT_X), XT_y)

    y_hat = np.dot(X, w_d)
    p[d - 1, :] = y_hat

    rel_err = np.linalg.norm(y - y_hat) / np.linalg .norm(y)
    print(f"d={d}:␣relative␣error␣=␣{rel_err*100:.3f}%")

# Plot the degree 1 surface
Z1 = p[0,:].reshape(data['x1'].shape)
ax = plt.axes(projection='3d')
ax.scatter(x1,x2,y) 
ax.plot_surface(data['x1'],data['x2'],Z1,color='orange') 
plt.show()

# Plot the degree 2 surface
Z2 = p[1,:].reshape(data['x1'].shape)
ax = plt.axes(projection='3d')
ax.scatter(x1,x2,y) 
ax.plot_surface(data['x1'],data['x2'],Z2,color='orange') 
plt.show()

# Plot the degree 3 surface
Z3 = p[2,:].reshape(data['x1'].shape)
ax = plt.axes(projection='3d')
ax.scatter(x1,x2,y) 
ax.plot_surface(data['x1'],data['x2'],Z3,color='orange') 
plt.show()