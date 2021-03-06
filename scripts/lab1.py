#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def newtonMethod(x_0, f, f_x, eps = 0.001):
    x_next = x_0 - f(x_0) / f_x(x_0)
    values_trace = [f(x_next)]

    while abs(x_0 - x_next) > eps:
        x_0, x_next = x_next, x_0
        x_next = x_0 - f(x_0) / f_x(x_0)
        values_trace.append(f(x_next))

    return x_next, values_trace

def dichotomyMethod(x_l, x_r, f, eps = 0.001):
    values_trace = []

    while x_r - x_l > eps:
        x_mid = (x_l + x_r) / 2.
        if f(x_mid)*f(x_l) < 0:
            x_r = x_mid
        elif f(x_mid)*f(x_r) <= 0:
            x_l = x_mid
        else:
            return None, None
        values_trace.append(f(x_mid))

    return (x_l + x_l) / 2., values_trace

def deleteSimilar(array, value, eps = 0.001):
    filteredArray = []

    for x in array:
        if abs(x - value) > eps:
            filteredArray.append(x)

    return filteredArray

def main():

    f = lambda x: x**3 + x - 1
    f_x = lambda x: 3*x**2 + 1

    #|f(x)|, n=2, alpha=1
    x_opt, iterationsNewthon = newtonMethod(1., f, f_x)
    print('Root found by Newthon method = {}'.format(x_opt))
    x_opt, iterationsDich = dichotomyMethod(0.5, 1., f)
    print('Root found by dichotomy method = {}'.format(x_opt))

    plt.xlabel('Number of iterations')
    plt.ylabel('$|f(x)|$')
    plt.plot(range(1, len(iterationsNewthon) + 1), np.abs(iterationsNewthon), \
        'b-o', label='Newthon method')
    plt.plot(range(1, len(iterationsDich) + 1), np.abs(iterationsDich), 'g-o', \
        label='Dichotomy method')
    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('../pictures/lab1_convergence.png', format = 'png', dpi = 200)

    #x(alpha), n=2,4,6
    f = lambda x, n, alpha: x**(n + 1) + x - alpha
    f_x = lambda x, n, alpha: (n + 1)*x**n + 1
    alphaGrid = np.linspace(0., 10., 100)
    colors = ['r', 'g', 'b', 'c']

    plt.clf()
    plt.xlabel('$\\alpha$')
    plt.ylabel('$x$')

    for i in range(1, 4):
        roots = [newtonMethod(alpha, lambda x: f(x, i*2, alpha), \
            lambda x: f_x(x, i*2, alpha))[0] for alpha in alphaGrid]
        plt.plot(alphaGrid, roots, colors[i-1] + '-', label='$N='+str(i*2)+'$')

    plt.plot(alphaGrid, np.power(alphaGrid, [1./3]*len(alphaGrid)), \
        colors[3] + '--', label='$\\alpha^{1/3}$')

    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('../pictures/lab1_roots.png', format = 'png', dpi = 200)

    plt.clf()
    plt.xlabel('$\\alpha$')
    plt.ylabel('$x$')

    f = lambda x, alpha: x**5 - alpha*x**4 + 2*x**3 - 2*alpha*x**2 \
        + (alpha**2 + 1)*x - alpha
    f_x = lambda x, alpha: 5*x**4 - 4*alpha*x**3 + 6*x**2 - \
        4*alpha*x + alpha**2 + 1.

    mid = np.power(alphaGrid[-1], 1./3.)
    initialPoints = [np.random.uniform(mid - 100., mid + 100.) for _ in range(100)]
    roots = [newtonMethod(x0, lambda x: f(x, alphaGrid[-1]), \
        lambda x: f_x(x, alphaGrid[-1]))[0] for x0 in initialPoints]

    filteredRoots = []
    while len(roots) != 0:
        filteredRoots.append(roots[0])
        roots = deleteSimilar(roots, filteredRoots[-1])

    alphaGrid = np.linspace(0., 10., 300)
    filteredRoots = sorted(filteredRoots)
    xValues = np.array([[0.]*len(alphaGrid)]*len(filteredRoots))

    for i in reversed(range(len(alphaGrid))):
        newRoots = []
        for j in range(len(filteredRoots)):
            newRoots.append(newtonMethod(filteredRoots[j], \
                lambda x: f(x, alphaGrid[i]), lambda x: f_x(x, alphaGrid[i]))[0])
            xValues[j][i] = newRoots[j]
        filteredRoots = newRoots

    for i, values in enumerate(xValues):
        plt.plot(alphaGrid, values, colors[0] + '-')

    plt.grid()
    plt.savefig('../pictures/lab1_bifurcation.png', format = 'png', dpi = 200)

if __name__ == '__main__':
    main()
