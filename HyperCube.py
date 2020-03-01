import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')

#HyperCube
corner = [np.array([(-1)**n, (-1)**m, (-1)**i, (-1)**j]) for j in range(2) for i in range(2) for m in range(2) for n in range(2)]

#hyperTriangle
#corner = [np.array([1, 1, 1, -1/math.sqrt(5)]), np.array([1, -1, -1, -1/math.sqrt(5)]), np.array([-1, 1, -1, -1/math.sqrt(5)]), np.array([-1, -1, 1, -1/math.sqrt(5)]), np.array([0, 0, 0, math.sqrt(5) - 1/math.sqrt(5)])]

def matrix_comp(M):
    if len(M) == 1:
        return M[0]
    else:
        return M[0].dot(matrix_comp([M[i] for i in range(1, len(M))]))
"""
theta = math.pi/2
R = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(theta), math.sin(theta)], [0, 0, -math.sin(theta), math.cos(theta)]])

for x in corner:
    x_ = np.dot(R, x)
    for y in corner:
        y_ = np.dot(R, y)
        z = x-y
        if np.linalg.norm(z) <= 2:
            plt.plot([x_[0]+x_[2]/3, y_[0]+y_[2]/3], [x_[1]+x_[2]/5, y_[1]+y_[2]/5], c = "blue")
plt.axis("scaled")
plt.show()"""

def animation(i):
    ax.clear()

    ax.axis('scaled')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    theta_xy = 0
    theta_xz = 0
    theta_xw = -i/20
    theta_yz = 0
    theta_yw = 0
    theta_zw = 0
    R_xy = np.array([[math.cos(theta_xy), math.sin(theta_xy), 0, 0], [-math.sin(theta_xy), math.cos(theta_xy), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R_xz = np.array([[math.cos(theta_xz), 0, math.sin(theta_xz), 0], [0, 1, 0, 0], [-math.sin(theta_xz), 0, math.cos(theta_xz), 0], [0, 0, 0, 1]])
    R_xw = np.array([[math.cos(theta_xw), 0, 0, math.sin(theta_xw)], [0, 1, 0, 0], [0, 0, 1, 0], [-math.sin(theta_xw), 0, 0, math.cos(theta_xw)]])
    R_yz = np.array([[1, 0, 0, 0], [0, math.cos(theta_yz), math.sin(theta_yz), 0], [0, -math.sin(theta_yz), math.cos(theta_yz), 0], [0, 0, 0, 1]])
    R_yw = np.array([[1, 0, 0, 0], [0, math.cos(theta_yw), 0, math.sin(theta_yw)], [0, 0, 1, 0], [0, -math.sin(theta_yw), 0, math.cos(theta_yw)]])
    R_zw = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(theta_zw), math.sin(theta_zw)], [0, 0, -math.sin(theta_zw), math.cos(theta_zw)]])
    R = matrix_comp([R_xw])

    camera = np.array([0, 0, 0, 2.5])

    def proj_3d(x):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).dot(x)

    def stereo_proj(x):
        d = camera[3]
        return np.array([[1/(d-x[3]), 0, 0, 0], [0, 1/(d-x[3]), 0, 0], [0, 0, 1/(d-x[3]), 0]]).dot(x)

    for x in corner:
        x_ = stereo_proj(np.dot(R, x))
        for y in corner:
            y_ = stereo_proj(np.dot(R, y))
            if np.linalg.norm(x-y) == 2:
                #plt.plot([x_[1] + x_[0] / 3, y_[1] + y_[0] / 3], [x_[2] + x_[0] / 3, y_[2] + y_[0] / 3], c="blue")
                #plt.plot([x_[0], y_[0]], [x_[1], y_[1]], [x[2], y[2]], c="blue")
                ax.plot([x_[0], y_[0]], [x_[1], y_[1]], [x_[2], y_[2]], c="blue") #stéréographique
    X = [stereo_proj(np.dot(R, x)) for x in corner]
    ax.scatter([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], c='blue')



fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1500)

ani = anim.FuncAnimation(fig, animation, interval=100, frames=300)
ani.save('HyperCube_3.mp4', writer=writer)
plt.show()