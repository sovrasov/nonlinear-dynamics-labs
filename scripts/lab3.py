#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def eulerMethod(f, a, b, f_0, step):
    numSteps = int((b - a) / step)

    xValues = [0.]*(numSteps + 1)
    tValues = [0.]*(numSteps + 1)
    xValues[0] = f_0
    tValues[0] = a

    for i in range(0, numSteps):
        xValues[i + 1] = xValues[i] + step*f(xValues[i], tValues[i])
        tValues[i + 1] = tValues[i] + step

    return tValues, xValues

def rungeKuttaMethod(f, a, b, f_0, step):
    numSteps = int((b - a) / step)

    xValues = [0.]*(numSteps + 1)
    tValues = [0.]*(numSteps + 1)
    xValues[0] = f_0
    tValues[0] = a
    step2 = step / 2

    for i in range(0, numSteps):
        k1 = f(xValues[i], tValues[i])
        k2 = f(xValues[i] + step2*k1, tValues[i] + step2)
        k3 = f(xValues[i] + step2*k2, tValues[i] + step2)
        k4 = f(xValues[i] + step*k3, tValues[i] + step)
        xValues[i + 1] = xValues[i] + step * (k1 + 2*k2 + 2*k3 +k4) / 6.
        tValues[i + 1] = tValues[i] + step

    return tValues, xValues

def plotError(steps, tValues, errValues, outputName, isLogYAxis=False):
    colors = ['r', 'g', 'b', 'c']

    plt.xlabel('t')
    plt.ylabel('error(t)')
    if isLogYAxis:
        plt.yscale('log')

    for i, step in enumerate(steps):
        plt.plot(tValues[i], errValues[i], \
            colors[i] + '-', label='Step =' + str(step))

    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig(outputName, format = 'png', dpi = 200)
    plt.clf()

def main():

    steps = [0.1, 0.01, 0.001]
    f = lambda x, t: x

    tValues = []
    errValues = []
    for i, step in enumerate(steps):
        timeGrid, xValues = eulerMethod(f, 0., 100., 1., step)
        errValues.append(np.abs(xValues - np.exp(timeGrid)))
        tValues.append(np.array(timeGrid))

    plotError(steps, tValues, errValues, '../pictures/lab3_exp_plus.png', True)

    f = lambda x, t: -x

    tValues = []
    errValues = []
    for i, step in enumerate(steps):
        timeGrid, xValues = eulerMethod(f, 0., 100., 1., step)
        errValues.append(np.abs(xValues - np.exp(-np.array(timeGrid))))
        tValues.append(np.array(timeGrid))

    plotError(steps, tValues, errValues, '../pictures/lab3_exp_minus.png', True)

    f = lambda x, t: np.array([x[1], -x[0]])

    tValues = []
    errValues = []
    for i, step in enumerate(steps):
        timeGrid, xValues = eulerMethod(f, 0., 20., np.array([1.,0.]), step)
        errValues.append(np.abs(np.array(xValues)[:,0] - np.cos(np.array(timeGrid))))
        tValues.append(np.array(timeGrid))

    plotError(steps, tValues, errValues, '../pictures/lab3_cons_system.png')

    f = lambda x, t: np.array([-x[1] - x[2], x[0] + 0.3*x[1], 0.3 + (x[0] - 5.7)*x[2]])

    tValues = []
    errValues = []
    for i, step in enumerate(steps):
        timeGrid, xValues = rungeKuttaMethod(f, 0., 100., np.array([1.,0., 1.]), step)
        _, xValues2 = rungeKuttaMethod(f, 0., 100., np.array([1.,0., 1.]), step / 2.)
        errValues.append(np.sum(np.abs(np.array(xValues) - np.array(xValues2)[::2]), axis=1))
        tValues.append(np.array(timeGrid))

    plotError(steps, tValues, errValues, '../pictures/lab3_rossler_system.png', True)

if __name__ == '__main__':
    main()
