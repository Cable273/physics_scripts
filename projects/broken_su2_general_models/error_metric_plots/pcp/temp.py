#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

x = np.load("./pcp,2ndOrder,fidelity,12.npy")
plt.plot(x)
plt.ylim((0.98,1))
plt.show()

