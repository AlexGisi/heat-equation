# Ref:
# https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
# https://www.math.ubc.ca/~peirce/M257_316_2012_Lecture_8.pdf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plate_length = 50
max_iter_time = 500

alpha = 1.0
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# The solution grid: a square plate.
u = np.empty((max_iter_time, plate_length, plate_length))

# Initial condition everywhere inside the grid.
u_init = 0.0

# Boundary conditions.
u_top = 100.0
u_left = 100.0
u_bottom = 0.0
u_right = 0.0

# Set the initial condition.
u.fill(u_init)

# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right


def calculate(u):
    for k in range(0, max_iter_time-1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k+1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j])\
                               + u[k][i][j]

    return u


def plotheatmap(u_k, k):
    plt.clf()

    plt.title(f"Temperature at t = {k * delta_t:.3f} unit time")
    plt.xlabel("x position")
    plt.ylabel("y position")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt


def animate(k):
    plotheatmap(u[k], k)


u = calculate(u)
anim = FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save(f'gen/heat_sim_{max_iter_time}.gif')
