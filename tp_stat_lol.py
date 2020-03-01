import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random

plt.style.use('dark_background')

n = 100
a = 10
dt = 20 / (10*n)
x = np.arange(0, a, a / 1000)
N = 10
X = [0 * np.copy(x) for _ in range(N)]
c = [np.zeros(100) for _ in range(N)]
for i, c_i in enumerate(c):
    for n, __ in enumerate(c_i):
        c_i[n] = random.choice([1, -1]) * random.random() / (n + 1) ** 1.8
for i, c_i in enumerate(c):
    for n, c_n in enumerate(c_i):
        X[i] = X[i] + c_n * np.sin((i + 1) * math.pi * x / a)

x_max = max([max(x) for x in X])
x_min = min([min(x) for x in X])
M = max([abs(x_min), abs(x_max)])


def animation(i):
    global dt, x, X, c
    ax.clear()

    X = [0 * np.copy(x) for _ in range(N)]
    for m, c_i in enumerate(c):
        for n, c_n in enumerate(c_i):
            X[m] = X[m] + c_n * np.sin((n + 1) * math.pi * x / a) * np.cos(-i * (n + 1) * dt) * np.exp(-i * (n + 1) * dt / 10)

    ax.set_ylim(-1.2 * M, 1.2 * M)

    for n in range(N):
        plt.plot(x, X[n])


"""
plt.plot(x,X)
plt.show()
"""

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1500)

ani = anim.FuncAnimation(fig, animation, interval=100, frames=500)
ani.save('tp_stat_lol_1.mp4', writer=writer)
plt.show()
