__author__ = "Alexandra Diem <alexandra@simula.no>"

from scipy.interpolate import interp1d
import numpy as np
from itertools import chain
import sys

import dolfin as df


def read_time_data(fname, data_unit=1, time_unit=1):
    """
    Read time data (csv) from file and load into Numpy array
    """
    data = np.loadtxt(fname, delimiter=',')
    t = data[:,0]*time_unit
    x = data[:,1]*data_unit
    f = interp1d(t, x, kind='linear', bounds_error=False, fill_value=x[0])
    return f


def write_file(f, u, label, t):
    df.set_log_level(40)
    f.write_checkpoint(u, label, t)
    df.set_log_level(30)


def read_file(f, u, label, i):
    df.set_log_level(40)
    f.read_checkpoint(u, label, i)
    df.set_log_level(30)
    return u


def mark_inlet(markers):
    from random import random

    # get list of all outer boundary elements
    marked_cells40 = df.SubsetIterator(markers, 40)
    marked_cells10 = df.SubsetIterator(markers, 10)

    # randomly mark cells as IN cells
    for cell in chain(marked_cells10, marked_cells40):
        if random() < 0.01:
            markers[cell] = 1


def print_time(t):
    print("t = {0: >#016.5f}".format(t), end='\r')
    sys.stdout.flush()


def periodic(t, T):
    while t > T:
        t -= T
    return t
