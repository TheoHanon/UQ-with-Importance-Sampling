import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from matplotlib.widgets import Slider



N = 20000
dt = .001
nStep = 1000
t = np.arange(nStep) * dt

var_x =  .5
var_yx = .5
var_y = var_x + var_yx

y = 0.0 # we assume mu_prop = mu_y = 0
var = 1 / (1/var_x + 1/var_yx)
mu_y = var/var_yx *y

var_prop = 2.0
mu_prop = 0.0

x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (N, 1)).reshape(-1)
# x0 = np.random.uniform(-0, 20, N)
# np.random.seed()


def p_x(x):
    return 1/np.sqrt(2*np.pi*var_x) * np.exp(-x**2/(2*var_x))

def p_yx(y, x):
    return 1/np.sqrt(2*np.pi*var_yx) * np.exp(-(y-x)**2/(2*var_yx))

def p_y(y):
    return 1/np.sqrt(2*np.pi*var_y) * np.exp(-y**2/(2*var_y))

def p_xy(x, y):
    return p_yx(y, x) * p_x(x) / p_y(y)

def q_t(x, t):
    var_t = np.exp(-2*t[:, np.newaxis]/var) * var_prop
    return 1/np.sqrt(2*np.pi*var_t) * np.exp(-(x - mu_y - np.exp(-t[:, np.newaxis]/var) * (mu_prop - mu_y))**2/(2*var_t))

def get_xt(x0, t):
    return np.exp(-t[:, np.newaxis]/var) * x0 + (1 - np.exp(-t[:, np.newaxis]/var)) * mu_y


def IP(alpha, x, q):
    f = lambda x: x**alpha
    p = p_xy(x, y)
    w =  (p / q) / np.sum(p / q)
    return np.sum(f(x) * w)

def moment(alpha):
    f = lambda x: x**alpha * p_xy(x, y)
    return quad(f, -np.inf, np.inf)[0]


def IP1_analytical(x0, t):
    xt = np.exp(-t/var) * x0 + (1 - np.exp(-t/var)) * mu_y
    var_t = np.exp(-2*t/var) * var_prop

    num = p_xy(xt, y)
    den = np.clip(1/np.sqrt(2*np.pi*var_t) * np.exp(-(xt- mu_y - np.exp(-t/var) * (mu_prop - mu_y))**2 / (2*var_t)), 1e0, None)
    w = num/den / np.sum(num/den)
    # print(np.max(w))
    return np.sum(xt * w)

def MSE(t, N_sample = 300000):
    var_t = np.exp(-2*t/var) * var_prop
    mu_y_t = mu_y - np.exp(-t/var) * (mu_prop - mu_y)

    x_sample = np.random.normal(mu_y_t, np.sqrt(var_t), N_sample)

    qt = 1/np.sqrt(2*np.pi*var_t) * np.exp(-(x_sample - mu_y_t)**2 / (2*var_t))
    p = p_xy(x_sample, y)

    return 1/N*np.mean((p/qt * x_sample)**2)

def exact_MSE(t):
    return 1/N * np.sqrt(var_prop)/(2*np.sqrt(2)*var) * np.exp(-t) * (1/var - np.exp(2*t)/(2*var_prop))**(-3/2)


def at(t):
    var_t = np.exp(-2*t/var) * var_prop
    return 1/var - 1/(2*var_t)

# anat = np.array([IP1_analytical(x0, ti) for ti in t])


# np.random.seed(10)

# mse = np.zeros((nStep, 50, 6))

# for i in range (50):
#     N_sample = [10, 100, 1000, 10000, 20000, 50000]
#     for j, n_sample in enumerate(N_sample):
#         x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n_sample, 1)).reshape(-1)
#         est = np.array([IP1_analytical(x0, ti) for ti in t])
#         mse[:, i, j] = est**2

# mse = np.mean(mse, axis = 1)


# fig, ax = plt.subplots(3, 2, figsize=(15, 6), sharex = True)
# ax = ax.ravel()
# for i, (axi, n_sample) in enumerate(zip(ax, N_sample)):
#     # x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n_sample, 1)).reshape(-1)
#     # est = np.array([IP1_analytical(x0, ti) for ti in t])

#     axi.fill_between(t, 0, np.sign(at(t)), where=(np.sign(at(t)) > 0), color='green', alpha=0.3, label = r'$a(t) > 0$')
#     axi.fill_between(t, 0, np.abs(np.sign(at(t))), where=(np.sign(at(t)) < 0), color='red', alpha=0.3, label = r'$a(t) < 0$')

#     axi.set_ylim([0, mse[:, i].max()])
#     axi.set_xlim([t[0], t[-1]])
#     if i % 2 == 0 : axi.set_ylabel('MSE')
#     if i == len(N_sample) - 1 or i == len(N_sample) - 2: axi.set_xlabel('Time')
#     axi.plot(t, mse[:, i])
#     axi.legend()
#     axi.set_title(f"N = {n_sample}", fontsize = 12, fontweight = "bold")

# plt.suptitle(r"Comparison of MSE for different sample sizes $(\mu_y = 0, \mu_q = 0)$", fontsize = 15, fontweight = "bold")
    
# plt.show()











xt = get_xt(x0, t)
qt = q_t(xt, t)


estimators = np.array([IP(1, x, q) for x, q in zip(xt, qt)])
m1 = moment(1)

# Create the figure and subplots
fig, (ax_scatter, ax_estimator) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25, wspace=0.4)

# Scatter plot
idx_sort = np.argsort(xt[0, :])
scatter = ax_scatter.scatter(xt[0, idx_sort], qt[0, idx_sort], marker='D', color = "k", s = 3)
line, = ax_scatter.plot(xt[0, idx_sort], qt[0, idx_sort], color='k', label = r"$q_t(x_t)$")
line_p, = ax_scatter.plot(np.linspace(-2, 2, 1000), p_xy(np.linspace(-2, 2, 1000),y), color='r', label = r"$p_{x|y}(x|y=0)$")

ax_scatter.set_xlabel('x')
ax_scatter.set_ylabel('y')
ax_scatter.set_xlim([-2, 2])
ax_scatter.set_ylim([0, 2])
ax_scatter.set_title(f'Scatter plot at t={t[0]:.2f}')
ax_scatter.legend()
# Estimator plot

# ax_estimator.plot(t, np.abs(anat - m1)**2, color='b', label = 'Analytical')
# ax_estimator.vlines(t_star(var, var_prop), 0, (np.abs(estimators - m1)**2).max(), color='k', label = 't* = {:.2f}'.format(t_star(var, var_prop)))
# ax_estimator.plot(t, np.sign(at(t)), color='b', label = r'$a(t)$')
ax_estimator.fill_between(t, 0, at(t), where=(np.sign(at(t)) > 0), color='green', alpha=0.3, label = r'$a(t) > 0$')
ax_estimator.fill_between(t, 0, np.abs(at(t)), where=(np.sign(at(t)) < 0), color='red', alpha=0.3, label = r'$a(t) < 0$')
estimator_line, = ax_estimator.plot(t[0], np.abs(estimators[0] - m1)**2, color='k', label = 'Estimator')
ax_estimator.set_xlabel('Time')
ax_estimator.set_ylabel(r'$\ell_1$-error')
ax_estimator.set_title('Moment 1')
ax_estimator.set_xlim([t[0], t[-1]])
ax_estimator.set_ylim([(np.abs(estimators - m1)**2).min(), (np.abs(estimators - m1)**2).max()])


ax_estimator.legend()

estimator_scatter = ax_estimator.scatter(t[0], np.abs(estimators[0] - m1)**2, color='r', s = 20, marker = 'D')


# Slider setup
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(t) - 1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    sorted_indices = np.argsort(xt[idx, :])
    scatter.set_offsets(np.c_[xt[idx, sorted_indices], qt[idx, sorted_indices]])
    ax_scatter.set_title(f'Scatter plot at t={t[idx]:.2f}')

    line.set_xdata(xt[idx, sorted_indices])
    line.set_ydata(qt[idx, sorted_indices])

    line_p.set_xdata(np.linspace(-2, 2, 1000))
    line_p.set_ydata(p_xy(np.linspace(-2, 2, 1000), y))


    estimator_line.set_xdata(t[:idx])
    estimator_line.set_ydata(np.abs(estimators[:idx] - m1)**2)

    estimator_scatter.set_offsets(np.c_[t[idx], np.abs(estimators[idx] - m1)**2])
    
    
    fig.canvas.draw_idle()

slider.on_changed(update)


plt.show()