# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:54:10 2023

@author: vadan
"""

import matplotlib.pyplot as plt
import numpy as np
import math as m


def create_plot(data_input1, name1, data_input2, name2,
                data_input3, name3, data_input4, name4):
    """
    plot 4 data sets
    -------
    None.

    """

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_title('Binomial Distributions → Gaussian')
    ax.set_xlabel('$n/N$')
    ax.set_ylabel('$\Omega(N,n)$ / $\Omega_{max}$')
    ax.scatter(data_input1[:, 0], data_input1[:, 1], s=30, c='r', label=name1)
    ax.scatter(data_input2[:, 0], data_input2[:, 1], s=15, label=name2)
    ax.plot(data_input3[:, 0], data_input3[:, 1],
            c='pink', alpha=0.5, label=name3)
    ax.plot(data_input4[:, 0], data_input4[:, 1],
            c='navy', alpha=0.5,  label=name4)
    plt.legend(loc='upper left',
               borderaxespad=0.5, fontsize='7')
    plt.tight_layout()
    plt.savefig('graph.png', dpi=1000)
    plt.show()

    return None


def max_bin(N_input):
    max_bin = m.factorial(N_input) / (m.factorial(N_input/2))**2
    return max_bin


def nCr(n_input, r_input):
    ans = m.comb(n_input, r_input)
    return ans


def data_compute(N):

    n_vals = np.array(range(0, N+1))
    # print(n_vals)
    nbyNvals = n_vals / N

    nCr_vals = np.empty(0)
    for n in n_vals:
        nCr_val = nCr(N, n)
        nCr_vals = np.append(nCr_vals, nCr_val)
    # print(nCr_vals)
    nCr_max = max_bin(N)
    nCrbymax_vals = nCr_vals/nCr_max

    data_out = np.column_stack((nbyNvals, nCrbymax_vals))

    return data_out


def gauss_compute(N_input):
    N = float(N_input)
    n_vals = np.linspace(0, N, 777)
    nbyNvals = n_vals / N
    s_vals = n_vals - N/2
    approx_vals = np.empty(0)
    for s in s_vals:
        approx = m.exp(-(2*(s)**2/N_input))
        approx_vals = np.append(approx_vals, approx)
    data_out = np.column_stack((nbyNvals, approx_vals))

    return data_out


# %% Main
data_10 = data_compute(10)
data_100 = data_compute(100)
data_g10 = gauss_compute(10)
data_g100 = gauss_compute(100)

create_plot(data_10, 'N = 10', data_100, 'N = 100',
            data_g10, 'Gauss Approximation for N=10', data_g100,
            'Gauss Approximation for N=100')
