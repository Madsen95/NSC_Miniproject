# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:53:11 2022

@author: bjark
"""
import numpy as np
import os
import time
import mandelbrot_functions as mf
import matplotlib.pyplot as plt

# sizes = [256, 516, 1024, 2048, 4096, 8192]

itr = 3
# times_20 = []
# print('(2.0), no worker size')
# for i, size in enumerate(sizes):
#     c = mf.create_mesh(size)
#     td = 0
#     for j in range(itr):
#         t0 = time.time()
#         mf.mandelbrot_gpu(c, pl_id=1, dev_id=0)
#         t1 = time.time()
#         td += t1 - t0
#     print(size, td/itr)
#     times_20.append(td/itr)

# plt.plot(sizes, times_20)
# plt.xlabel('Points')
# plt.ylabel('Execution time')
# plt.xticks(sizes, rotation=45)
# plt.tight_layout()
# plt.savefig('cpu_benchmark.pdf', dpi=(200))
# plt.show()

sizes = [1024, 2048, 4096, 6144, 8192, 10240, 11264]

c = mf.create_mesh(1024)
mf.mandelbrot_gpu(c, pl_id=2, dev_id=0)

times_20 = []
print('(2.0), no worker size')
for i, size in enumerate(sizes):
    c = mf.create_mesh(size)
    td = 0
    for j in range(itr):
        t0 = time.time()
        mf.mandelbrot_gpu(c, pl_id=2, dev_id=0)
        t1 = time.time()
        td += t1 - t0
    print(size, td/itr)
    times_20.append(td/itr)
    
times_20w = []
print('(2.0), 32x32 worker size')
for i, size in enumerate(sizes):
    c = mf.create_mesh(size)
    td = 0
    for j in range(itr):
        t0 = time.time()
        mf.mandelbrot_gpu(c, pl_id=2, dev_id=0, group_size=(32, 32))
        t1 = time.time()
        td += t1 - t0
    print(size, td/itr)
    times_20w.append(td/itr)
    
times_30 = []
print('(3.0), no worker size')
for i, size in enumerate(sizes):
    c = mf.create_mesh(size)
    td = 0
    for j in range(itr):
        t0 = time.time()
        mf.mandelbrot_gpu(c, pl_id=3, dev_id=0)
        t1 = time.time()
        td += t1 - t0
    print(size, td/itr)
    times_30.append(td/itr)

times_30w = []
print('(3.0), 16x16 worker size')
for i, size in enumerate(sizes):
    c = mf.create_mesh(size)
    td = 0
    for j in range(itr):
        t0 = time.time()
        mf.mandelbrot_gpu(c, pl_id=3, dev_id=0, group_size=(16, 16))
        t1 = time.time()
        td += t1 - t0
    print(size, td/itr)
    times_30w.append(td/itr)

plt.plot(sizes, times_20, label='GeForce GTX 1050 Ti (None)')
plt.plot(sizes, times_20w, label='GeForce GTX 1050 Ti (32x32)')
plt.plot(sizes, times_30, label='HD Graphics 630 (None)')
plt.plot(sizes, times_30w, label='HD Graphics 630 (16x16)')
plt.xlabel('Points')
plt.ylabel('Execution time')
plt.xticks(sizes, rotation=45)
plt.tight_layout()
plt.legend()
plt.savefig('gpu_benchmark.pdf', dpi=(200))
plt.show()