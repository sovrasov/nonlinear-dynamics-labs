#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from lab3 import rungeKuttaMethod


def naive_solver(initial_point, max_time, inc_rules, dec_rules):
    pass


def gillespie_solver(initial_point, max_time, inc_rules, dec_rules):
    assert len(inc_rules) == len(dec_rules)
    assert len(initial_point) == len(dec_rules)
    time = 0.
    x = initial_point.copy()
    x_s = [initial_point]
    times = [0.]
    while time < max_time:
        v_plus = []
        for i, rule in enumerate(inc_rules):
            v_plus.append(rule(x[i]))
        v_minus = []
        for i, rule in enumerate(dec_rules):
            v_minus.append(rule(x[i]))
        a_0 = np.sum(np.array(v_plus + v_minus))
        if a_0 <= 0:
            break
        probs = np.array(v_plus + v_minus) / a_0
        r1 = np.random.random() + 1e-12
        tao = np.log(1 / r1) / a_0
        time += tao
        idx = np.random.choice(2*len(initial_point), p=probs)
        if idx >= len(inc_rules):
            x[idx % len(inc_rules)] -= 1
        else:
            x[idx] += 1
        x_s.append(x.copy())
        times.append(time)

    return times, x_s

def main():

    np.random.seed(100)

    t_r, x_r = rungeKuttaMethod(lambda x, t: -x, 0., 10., 100., 1e-4)

    t, x = gillespie_solver([100], 10, [lambda x: 0], [lambda x: x])

    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.plot(t, np.array(x).reshape(-1), 'b-o', markersize=2, label='Gillespie solution')
    plt.plot(t_r, x_r, 'r-', label='Mean solution')
    plt.grid()
    plt.legend()
    plt.savefig('../pictures/lab7_eq_1.pdf', format = 'pdf')

    #t, x = gillespie_solver([100, 0], 100, [lambda x: 0], [lambda x: x])


if __name__ == '__main__':
    main()
