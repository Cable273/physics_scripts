#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

zero = np.load("./pcp,0thOrderErrors,14.npy")
one = np.load("./pcp,1stOrderErrors,14.npy")
two = np.load("./pcp,2ndOrderErrors,14.npy")
print(zero)
print(one)
print(two)

