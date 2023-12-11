import numpy as np

from scipy.integrate import cumtrapz

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/28766692/how-to-find-the-intersection-of-two-graphs/28766902#28766902

k = 1.38065e-23  # J/mol Boltzmann constant


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


def get_roots_bis(L, N, step=0.0001):
    alphas = np.arange(0, N, step=step)[1:]

    f = alphas

    g = L / np.tan(alphas)

    plt.plot(alphas, f, "-")
    plt.plot(alphas, g, "-")

    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    # remove one every other idx
    idx = idx[::2]
    plt.plot(alphas[idx], f[idx], "ro")
    plt.show()
    roots = alphas[idx]
    return roots


def analytical_expression_fractional_release(t, P_0, D, S, V, T, A, l):
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


# FIXME this doesn't work
# taken from https://mooseframework.inl.gov/TMAP8/verification/ver-1a.html#longhurst1992verification
def analytical_expression_cumulative_release(t, D, S, V, T, A, l):
    k = 1.38065e-23  # J/mol Boltzmann constant
    phi = 1 / (k * T * S)
    L = l * A / (V * phi)
    roots = get_roots_bis(L=L, N=2000, step=1e-3)
    roots = roots[:, np.newaxis]
    sec = 1 / np.cos(roots)
    summation = (2 * L * sec - np.exp(-(roots**2) * D * t / l**2)) / (
        L * (L + 1) + roots**2
    )
    last_term = summation[-1]
    summation = np.sum(summation, axis=0)
    print(summation[0])
    print(last_term / summation)
    cumulative_release = 1 - summation
    return cumulative_release


def analytical_expression_flux(t, P_0, D, S, V, T, A, l):
    k = 1.38065e-23  # J/mol Boltzmann constant
    L = S * T * A * k / V

    roots = get_roots(L=L, l=l, N=1e7, step=1)
    roots = roots[:, np.newaxis]

    summation = (np.exp(-(roots**2) * D * t) * roots) / (
        (l * (roots**2 + L**2) + L) * np.sin(roots * l)
    )
    last_term = summation[-1]
    summation = np.sum(summation, axis=0)

    print(last_term / summation)
    flux = 2 * S * P_0 * L * D * summation
    return flux


def cumulative_flux(t, P_0, D, S, V, T, A, l):
    flux = analytical_expression_flux(t, P_0, D, S, V, T, A, l)

    cumulative_flux = cumtrapz(flux, t, initial=0)
    initial_quantity = P_0 * V / k / T

    return cumulative_flux * A / initial_quantity


if __name__ == "__main__":
    T = 2373
    times = np.linspace(0, 45, 1000)
    # FR = analytical_expression_fractional_release(
    #     t=times,
    #     P_0=1e6,
    #     D=2.6237e-11,
    #     S=7.244e22 / T,
    #     V=5.20e-11,
    #     T=T,
    #     A=2.16e-6,
    #     l=3.3e-5,
    # )
    cum_flux = cumulative_flux(
        t=times,
        P_0=1e6,
        D=2.6237e-11,
        S=7.244e22 / T,
        V=5.20e-11,
        T=T,
        A=2.16e-6,
        l=3.3e-5,
    )
    # flux = analytical_expression_flux(
    #     t=times,
    #     P_0=1e6,
    #     D=2.6237e-11,
    #     S=7.244e22 / T,
    #     V=5.20e-11,
    #     T=T,
    #     A=2.16e-6,
    #     l=3.3e-5,
    # )
    plt.plot(times, cum_flux)
    plt.show()
