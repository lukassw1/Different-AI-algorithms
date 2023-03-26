from math import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator


def f_value(x):
    return (x**4)/4


def g_value(x: np.array):
    x1 = x[0]
    x2 = x[1]
    power1 = -(x1**2)-(x2**2)
    power2 = -(x1+1.5)**2-(x2-2)**2
    return 2 - np.exp(power1) - 0.5 * np.exp(power2)


def f_gradeint(x):
    return x**3


def g_gradeint(x: np.array):
    x1 = x[0]
    x2 = x[1]
    power1 = -(x1**2)-(x2**2)
    power2 = -(x1+1.5)**2-(x2-2)**2
    val1 = 2*x1*exp(power1) + (x1+1.5) * exp(power2)
    val2 = 2*x2*exp(power1) + (x2-2) * exp(power2)
    return np.array([val1, val2])


def plot(stops, beta):
    x = np.linspace(-5, 5, 1000)
    y = f_value(x)
    fig, ax = plt.subplots()
    plt.ylabel("y-coordinate")
    plt.xlabel("x-coordinate")
    plt.title("F Function")
    ax.plot(x, y)
    ax.scatter(stops, [f_value(x) for x in stops], color="black", label=f"steps: {len(stops)-1}\nbeta: {beta}")
    ax.legend()
    plt.savefig("example_f.png")
    plt.show()


def plot_3d(stops, beta):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = g_value((X, Y))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.CMRmap, linewidth=0, antialiased=False, alpha=0.4)
    ax.set_zlim(0.8, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('x1 Label')
    ax.set_ylabel('x2 Label')
    ax.set_zlabel('g(x1,x2) Label')
    x1 = [x[0] for x in stops]
    y1 = [y[1] for y in stops]
    z1 = [g_value(v) for v in stops]
    ax.scatter3D(x1, y1, z1, color='green', antialiased=False, s=75, label=f"steps: {len(stops)}\nbeta: {beta}")
    ax.legend()
    plt.savefig("example_g.png")
    # plt.show()


def countourf_plot(stops, beta):
    X, Y = np.meshgrid(np.linspace(-4, 4, 256), np.linspace(-2, 4, 256))
    Z = g_value((X, Y))
    levels = np.linspace(Z.min(), Z.max(), 7)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels = levels, extend='both', alpha=0.5, cmap='Blues')
    x1 = [x[0] for x in stops]
    y1 = [y[1] for y in stops]
    z1 = [g_value((x, y)) for x in x1 for y in y1]
    ax.scatter(x1, y1, s=25, color = "green", label=f"steps: {len(stops)}\nbeta: {beta}")
    ax.legend()
    ax.set_xlabel('x1 Label')
    ax.set_ylabel('x2 Label')
    fig.colorbar(cs)
    plt.savefig("example_g2d.png")
    plt.show()


def main():
    # plot(gradient_fall(f_gradeint_value, 0.5, 0.4), 0.4)
    # plot_3d(gradient_fall(g_gradeint_value, np.array([-3, 3]), 0.5), 0.5)
    # countourf_plot(gradient_fall(g_gradeint_value, np.array([-3, 3]), 0.5), 0.5)
    pass

if __name__ == "__main__":
    main()
