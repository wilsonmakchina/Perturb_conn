import numpy as np
from matplotlib import pyplot as plt

# we're going to find the matrix A from the equation B=AD where A is low rank
# by adding dynamics noise - noise that enters the computation itself - you can solve for A

# for simplicity there is technically no dynamics here. But you can imagine each column of D being a measurement of the
# neural activity of a set of neurons from a worm. Each column of B is the neural activity at the next time point,
# paired with the measurement in D.
# Each column is taken from a different animal, so we can be confident the measurements are independent. This lets us
# fit our dynamics matrix A without fancy math to deal with the correlations over time within a single worm

# We have access to D and B "corrupted" by noise, but this actually improves our prediction of A because it decorrelates
# the neurons

random_seed = 0
rng = np.random.default_rng(0)
dynamics_dim = 5  # the number of neurons
num_time = 10000  # each example is taken from one animal
noise_std = 0.5
dynamics_noise = noise_std * rng.standard_normal((dynamics_dim, num_time))
D_init = rng.standard_normal(dynamics_dim)

# make the true A
# A_true = rng.standard_normal((dynamics_dim, 1)) * rng.standard_normal((1, dynamics_dim))
A_true = rng.standard_normal((dynamics_dim, dynamics_dim))
A_true[1, :] = A_true[0, :].copy()
D_init[1] = D_init[0].copy()
eig_vals = np.linalg.eigvals(A_true)
A_true /= np.max(np.abs(eig_vals))

# Define D as a linear regression
D_no_noise = rng.standard_normal((dynamics_dim, num_time))
D_with_noise = D_no_noise + dynamics_noise

# calculate B
B_no_noise = A_true @ D_no_noise
B_with_noise = A_true @ D_with_noise

# now try to calculate A with and without noise
A_hat_no_noise = np.linalg.lstsq(D_no_noise.T, B_no_noise.T, rcond=None)[0].T
A_hat_with_noise = np.linalg.lstsq(D_with_noise.T, B_with_noise.T, rcond=None)[0].T

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(A_true)
plt.subplot(1, 3, 2)
plt.imshow(A_hat_no_noise)
plt.subplot(1, 3, 3)
plt.imshow(A_hat_with_noise)

plt.show()

# Define D as a dynamical system
D_no_noise = np.zeros((dynamics_dim, num_time))
D_no_noise[:, 0] = D_init
D_with_noise = np.zeros((dynamics_dim, num_time))
D_with_noise[:, 0] = D_init

for t in range(1, num_time):
    D_no_noise[:, t] = A_true @ D_no_noise[:, t-1]
    D_with_noise[:, t] = A_true @ D_no_noise[:, t-1] + dynamics_noise[:, t]

# now try to calculate A with and without noise
A_hat_no_noise = np.linalg.lstsq(D_no_noise[:, 0:-1].T, D_no_noise[:, 1:].T, rcond=None)[0].T
A_hat_with_noise = np.linalg.lstsq(D_with_noise[:, 0:-1].T, D_with_noise[:, 1:].T, rcond=None)[0].T

plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(D_no_noise)
plt.subplot(3, 1, 2)
plt.imshow(D_with_noise)
plt.subplot(3, 1, 3)
plt.imshow(dynamics_noise)

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(D_no_noise @ D_no_noise.T)
plt.subplot(2, 1, 2)
plt.imshow(D_with_noise @ D_with_noise.T)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(A_true)
plt.subplot(1, 3, 2)
plt.imshow(A_hat_no_noise)
plt.subplot(1, 3, 3)
plt.imshow(A_hat_with_noise)

plt.show()

