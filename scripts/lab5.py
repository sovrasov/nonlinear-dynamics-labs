#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Pool
import dill
import numpy as np
import matplotlib.pyplot as plt
from lab4 import getGUE
from lab3 import rungeKuttaMethod

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

def launchRK(f, a, b, f_0, step, idx):
    return idx, rungeKuttaMethod(f, a, b, f_0, step)[-1][-1]

def getPropagator(f, T, size, step):
    U = [[]]*size

    '''
    if size > 5:
        pool = Pool(4)
        jobs = []
        argsList = []
        for i in range(size):
            psi0 = np.zeros((size, 1))
            psi0[i] = 1.
            job = apply_async(pool, launchRK, (f, 0., T, psi0, step, i))
            jobs.append(job)

        for job in jobs:
            idx, uCol = job.get()
            U[idx] = uCol
    else:
    '''
    for i in range(size):
        psi0 = np.zeros((size, 1))
        psi0[i] = 1.
        uCol = rungeKuttaMethod(f, 0., T, psi0, step)[1][-1]
        U[i] = uCol

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

def plotHist(data, xLabel, yLabel, fileName, color):
    colors = ['blue', 'green', 'red']
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.hist(data, 50, normed = True,
        facecolor = color, alpha = 0.75)

    plt.grid()
    plt.savefig(fileName, format = 'pdf')
    plt.clf()

def main():

    np.random.seed(10)

    size = 100
    H0 = getGUE(size)
    H1 = getGUE(size)
    f = lambda x, t: -(H0 + 0.1*np.sin(2.*np.pi*t)*H1)*x * 1j

    epsPhi = []
    epsMu = []

    steps = np.logspace(-4, -1, 4, endpoint=True)
    '''
    for step in steps:
        U = getPropagator(f, 1., size, step)
        U2 = getPropagator(f, 1., size, step / 2.)

        values, vectors = np.linalg.eig(U)

        epsPhi.append(0.)
        for i in range(len(values)):
            phi2 = U2*vectors[:,i]
            epsPhi[-1] = max(epsPhi[-1], \
                np.linalg.norm(phi2 - np.matrix(vectors[:,i])*values[i]))

        epsMu.append( \
            np.linalg.norm(np.abs(values) - np.ones(len(values)), np.inf))
    plotError(steps, epsMu, \
                r'$\varepsilon_{\mu}$', '../pictures/lab5_eigvals_error.pdf')
    plotError(steps, epsPhi, \
            r'$\varepsilon_{\varphi}$', '../pictures/lab5_eigvecs_error.pdf')
    '''

    size = 100
    nImpls = 120
    fValues = [0, 0.01, 0.1]

    energySetsToPlot = []
    spacingSetsToPlot = []

    for F in fValues:
        allEnergies = []
        allSplits = []
        for i in range(nImpls):
            H0 = getGUE(size)
            H1 = getGUE(size)
            f = lambda x, t: -(H0 + H1*F*np.sin(2.*np.pi*t))*x * 1j

            U = getPropagator(f, 1., size, 0.005)
            _, vectors = np.linalg.eig(U)
            energysSet = []
            for j in range(len(vectors)):
                vector = np.matrix(vectors[:,j])
                energysSet.append(float(np.real(np.conj(vector.getT())*H0*vector)))

            energysSet = sorted(energysSet)
            allEnergies.append(energysSet)
            allSplits.append( \
                [energysSet[k+1] - energysSet[k] for k in range(len(energysSet) - 1)])

            print('f = {}, #impl: {} '.format(F, i))

        allEnergies = np.array(allEnergies).reshape(nImpls*len(allEnergies[0]))
        allSplits = np.array(allSplits).reshape(nImpls*len(allSplits[0]))
        allSplits /= np.mean(allSplits)
        allEnergies /= np.sqrt(size)

        energySetsToPlot.append(allEnergies)
        spacingSetsToPlot.append(allSplits)

        plotHist(allEnergies, r'$E$', r'$\rho(E)$',
            '../pictures/lab5_eigens_hist' + 'F=' + str(F) +'.pdf', 'red')

        plotHist(allSplits, r'$s$', r'$p(s)$',
            '../pictures/lab5_eigens_spacings_hist' + 'F=' + str(F) + '.pdf', 'green')

if __name__ == '__main__':
    main()
