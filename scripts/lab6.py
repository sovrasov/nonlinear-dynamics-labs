#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from lab1 import deleteSimilar

def computeImage(x_0, r, n):
    x = x_0
    lyapunov_characteristic = 0.
    for i in range(n):
        x = (r - x)*x
        logFx = np.log(abs(r - 2.*x))
        lyapunov_characteristic += logFx
    return x, lyapunov_characteristic / n

def filterArray(array, eps):
    filteredArray = array
    for i in range(len(array) / 2 + 1):
        filteredArray = deleteSimilar(filteredArray, array[i], eps)
        filteredArray.append(array[i])
    return filteredArray

def main():

    np.random.seed(100)
    nIterations = 1500
    nImpls = 120
    rValues = np.linspace(1e-3, 4, 1000)
    xStartValues = np.random.rand(nImpls)
    lConsts = np.zeros((len(rValues), 1))

    xFinishPoints = np.zeros((len(rValues), nImpls))
    for j in range(nImpls):
        for i, r in enumerate(rValues):
            xFinal, characteristic = computeImage(xStartValues[j], r, nIterations)
            xFinishPoints[i][j] = xFinal
            lConsts[i] += characteristic

    lConsts /= nImpls
    rChaos = rValues[np.argmax(lConsts >= 0.)]
    print('Edge of the chaos: r = {}'.format(rChaos))

    plt.xlabel('$r$')
    plt.ylabel('$x^*$')
    plt.ylim([-.5, np.amax(xFinishPoints)])

    for i, r in enumerate(rValues):
        valuesToPlot = np.array(filterArray(xFinishPoints[i], 1e-2))
        plt.plot([r], valuesToPlot.reshape((1, len(valuesToPlot))), \
            'b o', markersize = 1)

    plt.axvline(x=rChaos, color='b', linestyle='--')
    plt.grid()
    plt.savefig('../pictures/lab6_bifurcation_diagram.pdf', format = 'pdf')
    plt.clf()

    plt.xlabel('$r$')
    plt.ylabel(r'$\lambda$')
    plt.plot(rValues, lConsts, 'b-')
    plt.grid()
    plt.savefig('../pictures/lab6_lyapunov_characteristic.pdf', format = 'pdf')

if __name__ == '__main__':
    main()
