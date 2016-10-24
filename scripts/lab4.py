#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def getGOE(size):
    a = np.triu(np.matrix(np.random.normal(0., 1., (size, size))))
    return a + np.transpose(a)

def getGUE(size):
    a = np.matrix(np.random.normal(0., 1., (size, size)) + \
        1j*np.random.normal(0., 1., (size, size)))
    return 0.5*(a + a.getH())

def plotRhoHist(data, xLabel, yLabel, fileName, color):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    n, bins, patches = plt.hist(data, 50, normed = True,
        facecolor = color, alpha = 0.75)

    c = np.amax(n) / 2.
    f = np.vectorize(lambda x: c*np.sqrt(4. - x**2))
    lGrid = np.linspace(-2., 2., 100)
    plt.plot(lGrid, f(lGrid), color[0] + '-')

    plt.grid()
    plt.savefig(fileName, format = 'png', dpi = 200)
    plt.clf()

def plotHist():
    pass

def main():

    np.random.seed(10)

    size = 100
    nImpls = 300
    nBins = 100

    goeEigens = []
    gueEigens = []

    for i in range(nImpls):
        goeEigens.append(np.linalg.eigvalsh(getGOE(size)))
        gueEigens.append(np.real(np.linalg.eigvalsh(getGUE(size))))

    goeAllEigens = np.array(goeEigens).reshape(nImpls*size)
    gueAllEigens = np.array(gueEigens).reshape(nImpls*size)

    plotRhoHist(goeAllEigens / np.sqrt(size), r'$\lambda$', r'$\rho(\lambda)$',
        '../pictures/lab4_goe_eig_hist.png', 'green')
    plotRhoHist(gueAllEigens / np.sqrt(size), r'$\lambda$', r'$\rho(\lambda)$',
        '../pictures/lab4_gue_eig_hist.png', 'red')

    goeSplits = []
    for eigenSet in goeEigens:
        goeSplits.append([eigenSet[i+1] - eigenSet[i] for i in range(len(eigenSet) - 1)])
        goeSplits[-1] = np.array(goeSplits[-1]) / np.mean(goeSplits[-1])

    gueSplits = []
    for eigenSet in gueEigens:
        gueSplits.append([eigenSet[i+1] - eigenSet[i] for i in range(len(eigenSet) - 1)])
        gueSplits[-1] = np.array(gueSplits[-1]) / np.mean(gueSplits[-1])

    goeSplits = np.array(goeSplits).reshape(nImpls*len(goeSplits[0]))
    gueSplits = np.array(gueSplits).reshape(nImpls*len(gueSplits[0]))

    plt.xlabel(r'$\bar{s}$')
    plt.ylabel(r'$p(\bar{s})$')
    n, bins, patches = plt.hist(goeSplits, 100, normed = True,
        facecolor = 'green', alpha = 0.75)

    f = lambda x: x*np.exp(-np.pi / 4. * x**2)
    c = np.amax(n) / f(2. / np.pi) #np.pi / 2.
    f = np.vectorize(lambda x: c*x*np.exp(-np.pi / 4. * x**2))
    sGrid = np.linspace(np.amin(bins), np.amax(bins), 200)
    plt.plot(sGrid, f(sGrid), 'g-')

    print(np.amax(n), f(2. / np.pi))
    plt.grid()
    plt.savefig('../pictures/lab4_goe_split_hist.png', format = 'png', dpi = 200)
    plt.clf()

    plt.xlabel(r'$\bar{s}$')
    plt.ylabel(r'$p(\bar{s})$')
    n, bins, patches = plt.hist(gueSplits, 100, normed = True,
        facecolor = 'red', alpha = 0.75)

    f = lambda x: x**2*np.exp(-4. / np.pi * x**2)
    c = np.amax(n) / f(np.sqrt(np.pi) / 2.) #32. / np.pi**2
    f = np.vectorize(lambda x: c*x**2*np.exp(-4. /np.pi * x**2))
    sGrid = np.linspace(np.amin(bins), np.amax(bins), 200)
    plt.plot(sGrid, f(sGrid), 'r-')

    plt.grid()
    plt.savefig('../pictures/lab4_gue_split_hist.png', format = 'png', dpi = 200)
    plt.clf()

if __name__ == '__main__':
    main()
