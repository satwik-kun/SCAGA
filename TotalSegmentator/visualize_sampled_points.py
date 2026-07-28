import numpy as np
import matplotlib.pyplot as plt

EDGE_POINTS = 1382

points = np.load("sampled_points.npy") * 255

# Since shuffle is OFF:
edge = points[:EDGE_POINTS]
uniform = points[EDGE_POINTS:]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Edge-guided (green)
ax.scatter(
    edge[:,0],
    edge[:,1],
    edge[:,2],
    c='green',
    s=8,
    alpha=0.8,
    label='Edge-guided'
)

# Uniform (red)
ax.scatter(
    uniform[:,0],
    uniform[:,1],
    uniform[:,2],
    c='red',
    s=20,
    alpha=1.0,
    label='Uniform'
)

ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_zlim(0,255)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Mixed Gaussian Initialization")

ax.legend()

plt.tight_layout()
plt.show()