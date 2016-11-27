#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from lab4 import getGUE
from lab3 import rungeKuttaMethod

def getPropagator(f, T, size, step):
    U = []

    for i in range(size):
        psi0 = np.zeros((size, 1))
        psi0[i] = 1.
        uCol = rungeKuttaMethod(f, 0., T, psi0, step)[1][-1]
        U.append(uCol)

    return np.matrix(np.array(U).reshape(size, size)).getT()

def plotError(steps, errValues, yLabel, outputName):
    colors = ['r', 'g', 'b', 'c']

    plt.xlabel('$step$')
    plt.ylabel(yLabel)
    plt.yscale('log')
    plt.xscale('log')

    plt.plot(steps, errValues, colors[0] + '-o')

    plt.grid()
    plt.savefig(outputName, format = 'pdf')
    plt.clf()

def main():

    np.random.seed(10)

    size = 10
    H0 = getGUE(size)
    H1 = getGUE(size)
    f = lambda x, t: -(H0 + H1*0.1*np.sin(2.*np.pi*t))*x * 1j

    epsPhi = []
    epsMu = []

    steps = np.logspace(-4, -1, 5, endpoint=True)
    '''
    for step in steps:
        U = getPropagator(f, 1., size, step)
        U2 = getPropagator(f, 1., size, step / 2.)

        values, vectors = np.linalg.eig(U)

        epsPhi.append(-np.inf)
        for i in range(len(values)):
            phi2 = U2*vectors[:,i]
            epsPhi[-1] = max(epsPhi[-1], np.linalg.norm(phi2 - vectors[:,i]*values[i]))

        epsMu.append(np.linalg.norm(np.abs(values) - np.ones(len(values)), np.inf))

    plotError(steps, epsMu, r'$\varepsilon_{\mu}$', '../pictures/lab5_eigvals_error.pdf')
    plotError(steps, epsPhi, r'$\varepsilon_{\varphi}$', '../pictures/lab5_eigvecs_error.pdf')
    '''
    size = 3
    H0 = getGUE(size)
    H1 = getGUE(size)
    f = lambda x, t: -(H0 + H1*0.1*np.sin(2.*np.pi*t))*x * 1j

    U = getPropagator(f, 1., size, 1e-3)
    _, vectors = np.linalg.eig(U)
    energies = []
    for i in range(len(vectors)):
        vector = np.matrix(vectors[:,i])
        energies.append(float(np.real(vector.getT()*np.conj(H0*vector))))

    print(energies)


if __name__ == '__main__':
    main()
