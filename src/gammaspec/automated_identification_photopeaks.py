"""

functions included from 'Fearn et al, An Open-Source Iterative Python Module for the Automated Identification of Photopeaks in Photon Spectra, Radiation 2022; https://doi.org/10.3390/radiation2020014'
module main.py

"""

import numpy as np


def second_diff(N, m, z):
    """Finds second difference of spectrum, discrete analogue to second derivative."""

    # empty list with same size as data
    S = np.zeros(len(N))

    # loop over data points and for each take the second difference
    for i in range(len(N)):
        try:
            S[i] = N[i + 1] - 2 * N[i] + N[i - 1]
        # if at the ends of the array need to do different calculation to avoid error
        except IndexError:
            if i == 0:
                S[i] = N[i + 1] - 2 * N[i]
            else:
                S[i] = N[i - 1] - 2 * N[i]

    # smooth the second difference using Mariscotti's values
    for i in range(z):
        S = average(S, m)

    return S


def average(A, m):
    """Smooths input function by averaging neighboring values."""

    # define empty array same length as data
    B = np.zeros(len(A))

    # loop over data points and sum all points m either side of current one
    for i in range(len(A)):
        for j in range(i - m, i + m):
            try:
                B[i] += A[j]
            # if tries to go outside of boundaries of data, do nothing
            except IndexError:
                pass

    return B


def standard_dev_S(N, m, z):
    """Finds standard deviation of second difference."""
    # define empty array same size as data
    F = np.zeros(len(N))

    # loop over data points and take variance of the second difference
    for i in range(len(N)):
        try:
            F[i] = N[i + 1] + 4 * N[i] + N[i - 1]
        # need different calculation at ends of data array to avoid error
        except IndexError:
            if i == 0:
                F[i] = N[i + 1] + 4 * N[i]
            else:
                F[i] = N[i - 1] + 4 * N[i]

    # smoothing
    for i in range(z):
        F = average(F, m)

    # return square root of variance, std deviation
    return np.sqrt(F)