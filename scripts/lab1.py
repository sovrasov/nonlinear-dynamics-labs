#!/usr/bin/env python
# -*- coding: utf-8 -*-

def newtonMethod(x_0, f, f_x, eps = 0.001):
    x_next = x_0 - f(x_0) / f_x(x_0)

    iteration = 1
    while abs(x_0 - x_next) > eps:
        x_0, x_next = x_next, x_0
        x_next = x_0 - f(x_0) / f_x(x_0)
        iteration += 1

    return x_next, iteration

def dichotomyMethod(x_l, x_r, f, eps = 0.001):
    iteration = 0

    while x_r - x_l > eps:
        iteration += 1
        x_mid = (x_l + x_r) / 2.
        if f(x_mid)*f(x_l) < 0:
            x_r = x_mid
        elif f(x_mid)*f(x_r) <= 0:
            x_l = x_mid
        else:
            return None, None

    return (x_l + x_l) / 2., iteration

def main():

    #x_opt, iteration = newtonMethod(1.1, lambda x: x**5 - 1, lambda x: 5*x**4)
    x_opt, iteration = dichotomyMethod(0., 1.1, lambda x: x**5 - 1)
    print('Optimal point: {}, iteration {}'.format(x_opt, iteration))

if __name__ == '__main__':
    main()
