import numpy as np
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation 



N = 10000
dt = .01
nStep = 1000
t = np.arange(nStep) * dt
D = 1.0


mu = lambda x: -2*x
x0 = np.random.normal(-2, 10, N)


def langevin_diffusion(x0, D = 1):
    x = np.zeros((nStep, N))
    x[0] = x0
    for i in range(1, nStep):
        x[i] = x[i-1] + mu(x[i-1]) * dt + np.sqrt(2*D*dt) * np.random.normal(0, 1, N)
    return x

x = langevin_diffusion(x0, D)

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.hist(x[frame], bins=50, density=True)
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("x")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)   
    ax.set_ylabel("Probability density")
    
ani = FuncAnimation(fig, update, frames=nStep, interval=50)
plt.show()





# def Ito(x0, dt, n=2):
#     x = np.zeros((nStep, N))
#     x[0] = x0
#     for i in range(1, nStep):
#         x[i] = x[i-1] + n*mu(x[i-1]) * x[i]**(n-1) * dt + \
#             np.sqrt(2*D*dt) * np.random.randn()\
#             + np.sqrt(2*D)/2 * n*(n-1) * x[i]**(n-2) * dt
#     return x

# x = Ito(x0, dt)


# plt.plot(np.sqrt(x[-1, :]), x)
# plt.show()

