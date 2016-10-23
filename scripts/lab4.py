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

def plotHist(data, xLabel, yLabel, fileName, color):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    n, bins, patches = plt.hist(data, 50, normed = 1,
        facecolor = color, alpha = 0.75)
    plt.grid()
    plt.savefig(fileName, format = 'png', dpi = 200)
    plt.clf()

def main():

    np.random.seed(10)

    size = 100
    nImpls = 100
    nBins = 100

    goeEigens = []
    gueEigens = []

    for i in range(nImpls):
        goeEigens.append(np.sort(np.linalg.eig(getGOE(size))[0]))
        gueEigens.append(np.sort(np.real(np.linalg.eig(getGUE(size))[0])))

    goeAllEigens = np.array(goeEigens).reshape(nImpls*size)
    gueAllEigens = np.array(gueEigens).reshape(nImpls*size)

    plotHist(goeAllEigens / np.sqrt(size), r'$\lambda$', r'$\rho(\lambda)$',
        '../pictures/lab4_goe_eig_hist.png', 'green')
    plotHist(gueAllEigens / np.sqrt(size), r'$\lambda$', r'$\rho(\lambda)$',
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

    plt.xlabel('$s$')
    plt.ylabel(r'$p(s)$')
    n, bins, patches = plt.hist(goeSplits, 50, normed = 1,
        facecolor = 'green', alpha = 0.75)
    plt.grid()
    plt.savefig('../pictures/lab4_goe_split_hist.png', format = 'png', dpi = 200)
    plt.clf()

    plt.xlabel('$s$')
    plt.ylabel(r'$p(s)$')
    n, bins, patches = plt.hist(gueSplits, 50, normed = 1,
        facecolor = 'red', alpha = 0.75)
    plt.grid()
    plt.savefig('../pictures/lab4_gue_split_hist.png', format = 'png', dpi = 200)
    plt.clf()

#    print(np.amax(goeEigens), np.amin(goeEigens))

if __name__ == '__main__':
    main()
