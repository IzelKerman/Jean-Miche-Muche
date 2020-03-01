import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random

plt.style.use('dark_background')

def optimal_int(f, x_i, x_f, wanted_error, multiple = False):
    k = 0
    Dx = x_f - x_i
    int = f(x_f) - f(x_i) * Dx
    error = 100 * float(1 - int / 2)
    while (error > wanted_error and k <= 12) or k <= 5:
        k += 1
        int /= 2

        n = 2 ** (k - 1)
        dx = Dx / n
        for i in range(n):
            int += f(x_i + (i + 0.5) * dx) * dx / 2

        error = 100 * abs(1 - int / 2)

    if abs(int) < wanted_error:
        int = 0

    return int

def optimal_int_f_sin(f, n, x_i, x_f, wanted_error):
    k = 0
    Dx = x_f - x_i
    int = (f(x_f) * np.sin(n * 2 * np.pi * x_f / Dx) - f(x_i) * np.sin(n * 2 * np.pi * x_i / Dx)) * Dx
    error = 100 * float(1 - int / 2)
    while (error > wanted_error and k <= 12) or k <= 5:
        k += 1
        int /= 2

        m = 2 ** (k - 1)
        dx = Dx / m
        for i in range(m):
            int += f(x_i + (i + 0.5) * dx) * np.sin(n * 2 * np.pi * (x_i + (i + 0.5) * dx) / Dx) * dx / 2

        error = 100 * abs(1 - int / 2)

    if abs(int) < wanted_error:
        int = 0

    return int

def optimal_int_f_cos(f, n, x_i, x_f, wanted_error):
    k = 0
    Dx = x_f - x_i
    int = (f(x_f) * np.cos(n * 2 * np.pi * x_f / Dx) - f(x_i) * np.cos(n * 2 * np.pi * x_i / Dx)) * Dx
    error = 100 * float(1 - int / 2)
    while (error > wanted_error and k <= 12) or k <= 5:
        k += 1
        int /= 2

        m = 2 ** (k - 1)
        dx = Dx / m
        for i in range(m):
            int += f(x_i + (i + 0.5) * dx) * np.cos(n * 2 * np.pi * (x_i + (i + 0.5) * dx) / Dx) * dx / 2

        error = 100 * abs(1 - int / 2)

    if abs(int) < wanted_error:
        int = 0

    return int

def Fourier_series_coef(f, x_i, x_f, n_max, error = 0.001):
    T = (x_f + x_i)/2
    a = np.zeros(n_max + 1)
    a[0] = optimal_int(f, x_i, x_f, error) / (2 * T)
    b = np.zeros(n_max + 1)
    b[0] = 0
    for n in range(1, n_max+1):
        a[n] = optimal_int_f_cos(f, n, x_i, x_f, error) / T
        b[n] = optimal_int_f_sin(f, n, x_i, x_f, error) / T
        print("done for {}".format(n))
    return a, b

def fonction__(x):
    if isinstance(x, int) or isinstance(x, float):
        return 1
    else:
        return np.ones(len(x))

def fonction_(x):
    return x

def fonction(x):
    if isinstance(x, int) or isinstance(x, float):
        if x <= np.pi:
            return x
        else:
            return 2*np.pi - x
    else:
        X = np.copy(x)
        for x_i in X:
            if x_i <= np.pi:
                pass
            else:
                x_i = 2*np.pi - x_i
        return X

A, B = Fourier_series_coef(fonction, 0, 2*np.pi, 100)
x = np.arange(0, 2*np.pi, np.pi/2000)
F = 0 * np.copy(x)
for n in range(len(A)):
    F = F + A[n] * np.cos(n * x) + B[n] * np.sin(n * x)

M = max([abs(max(F)), abs(min(F))])

dt = 0.01
def animation(i):
    global dt, x, A, B
    ax.clear()

    F = 0 * np.copy(x)
    for n in range(len(A)):
        F = F + (A[n] * np.cos(n * x) + B[n] * np.sin(n * x)) * np.exp(-n * i * dt)

    ax.set_ylim(-1.2 * M, 1.2 * M)

    plt.plot(x, F)


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1500)

ani = anim.FuncAnimation(fig, animation, interval=100, frames=200)
ani.save('Fourier_3.mp4', writer=writer)
plt.show()