#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from lab1 import deleteSimilar

def computeImage(x_0, r, n):
    x = x_0
    for i in range(n):
        x = (r - x)*x
    return x

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
    xStartValues = np.random.rand(nImpls)*3.#np.linspace(1e-10, 2., nImpls)
    lConsts = np.zeros((len(rValues), 1))

    '''
    for i, x in enumerate(xStartValues):
        endPoints = []
        for r in rValues:
            endPoints.append(computeImage(x, r, 1000))
        plt.plot(rValues, endPoints, 'b o', markersize=2)
    '''

    xFinishPoints = np.zeros((len(rValues), nImpls))
    for i, r in enumerate(rValues):
        for j in range(nImpls):
            xFinal = computeImage(xStartValues[j], r, nIterations)
            xFinishPoints[i][j] = xFinal
            if xFinal != np.inf:
                multiplicator = np.log(abs(r - 2.*xFinal))
                if multiplicator != np.inf:
                    lConsts[i] += multiplicator
    lConsts /= nImpls
    #print(lConsts)
    print("Computations finished")
    rChaos = rValues[np.argmax(lConsts >= 0.)]
    print('Edge of the chaos: r = {}'.format(rChaos))

    plt.xlabel('$r$')
    plt.ylabel('$x^*$')
    plt.ylim([-.5, np.amax(xFinishPoints)])

    for i, r in enumerate(rValues):
        valuesToPlot = np.array(filterArray(xFinishPoints[i], 1e-2))
        plt.plot([r], valuesToPlot.reshape((1, len(valuesToPlot))), \
            'b o', markersize = 1)

    #for i in range(nImpls):
    #    plt.plot(rValues, xFinishPoints[:,i], 'b o', markersize = 1)

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
