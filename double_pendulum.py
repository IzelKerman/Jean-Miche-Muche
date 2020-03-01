import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random

plt.style.use('dark_background')
#plt.rcParams['figure.facecolor'] = '#36393F'

class Syst:
    def __init__(self, L, l, m1, m2, g):
        self.L = L
        self.l = l
        self.m1 = m1
        self.m2 = m2
        self.M = m1 + m2
        self.mu1 = m1 / self.M
        self.mu2 = m2 / self.M
        self.g = g


syst = Syst(2, 1, 1, 1, 10)


def f(t, X):
    """
    X = [theta, dt_theta, phi, dt_phi]
    """
    global syst
    theta, dt_theta, phi, dt_phi = X[0], X[1], X[2], X[3]

    alpha = 1 - syst.mu2 * math.cos(theta - phi) ** 2
    dtt_phi = (syst.L * dt_theta * dt_phi * math.sin(theta - phi) / (syst.l * alpha)) - (syst.g * math.sin(phi) / (syst.l * alpha)) + (syst.L * dt_theta * (dt_theta - dt_phi) * math.sin(theta - phi) / (syst.l * alpha))
    dtt_phi += -syst.mu2 * dt_phi * (dt_theta - dt_phi) * math.cos(theta - phi) * math.sin(theta - phi) / alpha + syst.g * math.cos(theta - phi) * math.sin(theta) / (syst.l * alpha)
    dtt_phi += syst.mu2 * dt_theta * dt_phi * math.sin(theta - phi) * math.cos(theta - phi) / alpha

    dtt_theta = syst.mu2 * syst.l * dt_phi * (dt_theta - dt_phi) * math.sin(theta - phi) / syst.L - syst.mu2 * syst.l * dtt_phi * math.cos(theta - phi) / syst.L - syst.g * math.sin(theta) / syst.L - syst.mu2 * syst.l * dt_theta * dt_phi * math.sin(theta - phi) / syst.L

    return np.array([dt_theta, dtt_theta, dt_phi, dtt_phi])


# le vol du prof !!!!!!!
def derivatives_rk_4(derivatives, x, y, h):
    k_1 = derivatives(x, y)
    k_2 = derivatives(x + h / 2, y + h / 2 * k_1)
    k_3 = derivatives(x + h / 2, y + h / 2 * k_2)
    k_4 = derivatives(x + h, y + h * k_3)
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6


def rk_4(derivatives, x, y, x_max, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < x_max:
        y = y + h * derivatives_rk_4(derivatives, x, y, h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

N = 10000
T = [0 for _ in range(N)]
X = [0 for _ in range(N)]
for i in range(N):
    T[i], X[i] = rk_4(f, 0, np.array([math.pi/2 - math.pi * i /10000000, 0, math.pi/2 - math.pi * i /10000000, 0]), 31, 0.01)

x_max = syst.L + syst.l
x_min = -syst.L - syst.l


def animation(i):
    global T, X
    ax.clear()

    #x = [0, syst.L * math.sin(X[0][3 * i][0]), syst.L * math.sin(X[0][3 * i][0]) + syst.l * math.sin(X[0][3 * i][2])]
    #y = [0, -syst.L * math.cos(X[0][3 * i][0]), -syst.L * math.cos(X[0][3 * i][0]) - syst.l * math.cos(X[0][3 * i][2])]

    ax.axis('scaled')
    ax.set_xlim(1.1 * x_min, 1.1 * x_max)
    ax.set_ylim(1.1 * x_min, 1.1 * x_max)

    #plt.plot(x, y)
    plt.plot([syst.L * math.sin(X[j][3 * i][0]) + syst.l * math.sin(X[j][3 * i][2]) for j in range(N)], [-syst.L * math.cos(X[j][3 * i][0]) - syst.l * math.cos(X[j][3 * i][2]) for j in range(N)])
    #x = [0, syst.L * math.sin(X[j][3 * i][0]), syst.L * math.sin(X[j][3 * i][0]) + syst.l * math.sin(X[j][3 * i][2])]
    #y = [0, -syst.L * math.cos(X[j][3 * i][0]), -syst.L * math.cos(X[j][3 * i][0]) - syst.l * math.cos(X[j][3 * i][2])]
    #plt.plot(x, y)

def animation2(i):
    global T, X
    ax.clear()

    #x = [0, syst.L * math.sin(X[0][3 * i][0]), syst.L * math.sin(X[0][3 * i][0]) + syst.l * math.sin(X[0][3 * i][2])]
    #y = [0, -syst.L * math.cos(X[0][3 * i][0]), -syst.L * math.cos(X[0][3 * i][0]) - syst.l * math.cos(X[0][3 * i][2])]

    ax.axis('scaled')
    ax.set_xlim(1.1 * x_min, 1.1 * x_max)
    ax.set_ylim(1.1 * x_min, 1.1 * x_max)

    #plt.plot(x, y)
    for j in range(N):
        plt.scatter([syst.L * math.sin(X[j][3 * i][0]) + syst.l * math.sin(X[j][3 * i][2])], [-syst.L * math.cos(X[j][3 * i][0]) - syst.l * math.cos(X[j][3 * i][2])])
        #x = [0, syst.L * math.sin(X[j][3 * i][0]), syst.L * math.sin(X[j][3 * i][0]) + syst.l * math.sin(X[j][3 * i][2])]
        #y = [0, -syst.L * math.cos(X[j][3 * i][0]), -syst.L * math.cos(X[j][3 * i][0]) - syst.l * math.cos(X[j][3 * i][2])]
        #plt.plot(x, y)


def animation3(i):
    global T, X
    ax.clear()

    ax.axis('scaled')
    ax.set_xlim(1.1 * x_min, 1.1 * x_max)
    ax.set_ylim(1.1 * x_min, 1.1 * x_max)

    # plt.plot(x, y)
    plt.scatter([syst.L * math.sin(X[j][3 * i][0]) + syst.l * math.sin(X[j][3 * i][2]) for j in range(N)], [-syst.L * math.cos(X[j][3 * i][0]) - syst.l * math.cos(X[j][3 * i][2]) for j in range(N)], c=[k for k in range(N)], cmap='plasma_r')


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# if you have ffmpeg instaled
# Writer = anim.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1000)

ani = anim.FuncAnimation(fig, animation3, interval=100, frames=1000)
# ani.save('double_10(best_but_even_better).mp4', writer=writer)
plt.show()







