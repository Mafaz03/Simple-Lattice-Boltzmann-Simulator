import numpy as np
from matplotlib import pyplot as plt

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

Nx = 400 # How many cells in x axis
Ny = 100 # How many cells in y axis

tau = 0.53 # Time scale
Nt = 6000 # TIme steps

# lattice speeds and weights
NL = 9 # 9 differentt velocities
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9 , 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# Initial Condition
F = np.ones((Ny, Nx, NL)) + (0.1 * np.random.randn(Ny, Nx, NL))
F[:, :, 3] = 2.3 # Every right node gets a nudge to the right 

cylinder = np.full((Ny, Nx), False) # False = NOT a Boundary, True = IS a Boundary
cylinder_radius = 13

# Defining Obsticle
for x in range(Nx):
    for y in range(Ny):
        if (distance(Nx//4, Ny//2, x, y) <= cylinder_radius):
            cylinder[y][x] = True 
cylinder[95:] = True     # Bottom Wall
cylinder[:5] = True      # Top Wall
# cylinder[:, :5] = True   # Left Wall
# cylinder[:, 395:] = True # Right Wall
plot_every = 1

for it in range(Nt):
    for i, cx, cy in zip(range(NL), cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)
    BoundaryF = F[cylinder, :]
    BoundaryF = BoundaryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # If particles are inside the Boundary, then make them go Opposite Direction

    # Fluid Variables
    rho = np.sum(F, axis = -1) # Density
    momentumx = np.sum(F * cxs, axis = -1) / rho
    momentumy = np.sum(F * cys, axis = -1) / rho

    F[cylinder, :] = BoundaryF
    momentumx[cylinder], momentumy[cylinder] = 0, 0

    # Collision
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
        Feq[:, :, i] = rho * w * (
            1 + 3 * (cx * momentumx + cy * momentumy) + 9 * (cx * momentumx + cy * momentumy)**2 / 2 - 3 * (momentumx ** 2 + momentumy ** 2) / 2
                              )
    F = F + -(1/tau) * (F-Feq)

    if it % plot_every == 0:
        plt.imshow(np.sqrt(momentumx ** 2 + momentumy ** 2))
        plt.title(it)
        plt.pause(0.00001)
        plt.cla()