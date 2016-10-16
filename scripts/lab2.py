#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from lab1 import newtonMethod

def main():

    f = lambda x, n, alpha: x**(n + 1) + x - alpha
    f_x = lambda x, n, alpha: (n + 1)*x**n + 1
    alphaGrid = np.linspace(10e-4, 10., 100)
    colors = ['r', 'g', 'b', 'c']

    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\tau$')

    for i in range(1, 4):
        roots = [newtonMethod(alpha, lambda x: f(x, i*2, alpha), \
            lambda x: f_x(x, i*2., alpha))[0] for alpha in alphaGrid]

        wValues = []
        for j, x0 in enumerate(roots):
            rootArg = (i*2.*(1. - x0 / alphaGrid[j]))**2 - 1.
            if rootArg > 0.:
                wValues.append(math.sqrt(rootArg))
            else:
                wValues.append(np.nan)

        tValues = []
        for j, w in enumerate(wValues):
            acosArg = 1./(i*2.)/(roots[j] / alphaGrid[j] - 1.) / w \
                if w is not np.nan else np.inf
            if acosArg < 1. and acosArg > -1.:
                tValues.append(math.acos(acosArg))
            else:
                tValues.append(np.nan)

        plt.plot(alphaGrid, tValues, colors[i-1]+'-o', label='$N='+str(i*2)+'$')

    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('../pictures/lab2_diagram.png', format = 'png', dpi = 200)

if __name__ == '__main__':
    main()
