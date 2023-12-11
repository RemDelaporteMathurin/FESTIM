import numpy as np


import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/28766692/how-to-find-the-intersection-of-two-graphs/28766902#28766902

N = 10000


def get_roots(L, l, N, step=0.0001):
    alphas = np.arange(0, N, step=step)[1:]

    f = alphas

    g = L / np.tan(alphas * l)

    # plt.plot(alphas, f, "-")
    # plt.plot(alphas, g, "-")

    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    # remove one every other idx
    idx = idx[::2]
    # plt.plot(alphas[idx], f[idx], "ro")
    # plt.show()
    roots = alphas[idx]
    return roots


# print(len(get_roots(N, step=0.0001)))
# plt.show()


def analytical_expression_fractional_release(t, P_0, D, S, V, T, A, l):
    k = 1.38065e-23  # J/mol Boltzmann constant
    L = S * T * A * k / V

    roots = get_roots(L=L, l=l, N=1e6, step=1)
    print(len(roots))
    roots = roots[:, np.newaxis]
    summation = np.exp(-(roots**2) * D * t) / (l * (roots**2 + L**2) + L)
    last_term = summation[-1]
    summation = np.sum(summation, axis=0)

    print(last_term / summation)
    pressure = 2 * P_0 * L * summation
    fractional_release = 1 - pressure / P_0
    return fractional_release


if __name__ == "__main__":
    T = 2373
    times = np.linspace(0, 100, 1000)
    FR = analytical_expression_fractional_release(
        t=times,
        P_0=1e6,
        D=2.6237e-11,
        S=7.244e22 / T,
        V=5.20e-11,
        T=T,
        A=2.16e-6,
        l=3.3e-5,
    )
    plt.plot(times, FR)
    plt.show()
