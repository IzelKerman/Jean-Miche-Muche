import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from matplotlib.collections import LineCollection

plt.rcParams['animation.ffmpeg_path'] = "C:/FFmpeg/ffmpeg-20200129-de1b2aa-win64-static/bin/ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = "C:/FFmpeg/ffmpeg-20200129-de1b2aa-win64-static/bin/ffmpeg.exe"
plt.style.use('dark_background')

T_1 = 3
T_2 = 0
a = 1e-2


def fct(X):
    F = np.zeros(len(X))
    for i, x in enumerate(X):
        if x < 0:
            F[i] = T_1
        elif x > 0:
            F[i] = T_2
        else:
            F[i] = (T_1 + T_2) / 2
    return F


def f1(X):
    return T_1 * (np.cos(X) + np.cos(2 * X) / 2 + np.cos(7 * X) / 10 + np.sin(3 * X) / 5)


# ce truc est nul
def f(fonction, X):
    delta = 0.2
    F = np.zeros(len(X))
    for i, x in enumerate(X):
        if x < X[0] + delta:
            F[i] = fonction(X[0] + delta)
        elif X[-1] - delta < x:
            F[i] = fonction(X[-1] - delta)
        else:
            F[i] = fonction(x)
    return F


def d_x(X):
    n = len(X) - 1
    return np.array([0] + [(X[i + 1] - X[i - 1]) / (2 * dx) for i in range(1, n, 1)] + [0])


"""
def dd_x(X):
    n = len(X) - 1
    return np.array([X[1] / dx] + [(X[2] - X[1]) / dx] + [(X[i + 1] - X[i - 1]) / (2 * dx) for i in range(2, n - 1, 1)] + [(X[n - 1] - X[n - 2]) / dx] + [- X[n - 1] / dx])"""


def dd_x(X):
    n = len(X) - 1
    return np.array([X[1] / dx] + [(X[i + 1] - X[i - 1]) / (2 * dx) for i in range(1, n, 1)] + [- X[n - 1] / dx])


def d(X):
    n = len(X)
    return np.array([(X[1] - X[0]) / (dx ** 2)] + [(X[i + 1] + X[i - 1] - 2 * X[i]) / (dx ** 2) for i in range(1, n - 1, 1)] + [(X[n] - X[n - 1]) / (dx ** 2)])


#x_0 = 2 * 0.0381348
#x_f = 2 * (math.pi - 1.14799)
x_0 = -10
x_f = 10
L = x_f - x_0
n = 500
dt = 1 / 4
dx = L / n
dx_max = dx / 10

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

X = np.linspace(x_0, x_f, n)
T = f1(X)
dT_x = d_x(T)
dT_xx = dd_x(T)

T_min = T.min()
T_max = T.max()
delta = max(abs(T_min), abs(T_max)) / 10
norm = plt.Normalize(T_min, T_max)


def animation(j):
    global T, dT_xx
    if j > 0:
        dT_xx = dd_x(d_x(T))
        # dT_xx = d(T)
        m = a * max(dT_xx.max(), abs(dT_xx.min())) * dt
        if max(m, dx_max) > dx_max:
            time = 0
            while time < dt:
                dT_xx = dd_x(d_x(T))
                # dT_xx = d(T)
                m = a * max(dT_xx.max(), abs(dT_xx.min())) * dt
                dt_tempo = dx_max / m
                T += a * dT_xx * dt_tempo
                time += dt_tempo
        else:
            T += a * dT_xx * dt
    ax.clear()

    points = np.array([X, T]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-2], points[2:]], axis=1)

    lc = LineCollection(segments, cmap='plasma', norm=norm)

    lc.set_array(0.5 * (T[:-2] + T[2:]))
    lc.set_linewidth(1)
    lc.set_antialiased(True)
    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)

    plt.ylim(T_min - delta, T_max + delta)
    plt.xlim(x_0, x_f)


Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1500)

ani = anim.FuncAnimation(fig, animation, interval=100, frames=600)
ani.save('test_5.mp4', writer=writer)
plt.show()
