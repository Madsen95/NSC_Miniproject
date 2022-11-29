# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:42:04 2022

@author: bjark
"""

import numpy as np
import os
import time
import mandelbrot_functions as mf
import matplotlib.pyplot as plt

gs1 = [None, (32, 32), (16, 16), (8, 8), (2, 2)]
itr = 3
points = 4096
c = mf.create_mesh(points)

times = []
for i, gs in enumerate(gs1):
    print(gs)
    td = 0
    for j in range(itr):
        t0 = time.time()
        mf.mandelbrot_gpu(c, pl_id=2, dev_id=0, group_size=gs)
        t1 = time.time()
        td += t1-t0
    times.append(td/itr)