"""
Mandelbrot set main run-file.
Set configuration as desored.
Run this file to test implementations.
"""
import time
import mandelbrot_functions as mf
import numpy as np
import os

"""Configuration:"""
points = 2**12 # Number of points (Keep the power to an even number!):
    # 2**A, where A = 6, 8, 10, 12
T = 2 # Threshold
I = 25 # Max iteration for mandelbrot
itr = 3 # Number of times to run each mandelbrot function
showImg = True # Plot each Mandelbrot function upon completion
saveImg = True # Save each plot upon completion
alpha = int(np.sqrt(points)) # Default blocksize for parallel and distributed

if __name__ == '__main__':
    print('Starting... Running each function', itr, 'time(s) with', points, 'points\n')
    c = mf.create_mesh(points)
    label = [] # Collection of methods used. This if for performance comparison
    times = [] # Collection of execution times, index identical to label
    
    # print('Naive Mandelbrot implementation...')
    # naive_time = []
    # for i in range(itr):
    #     t0 = time.time()
    #     heatmap_naive = mf.mandelbrot_naive(c, T, I, data_type=np.float64)
    #     naive_time.append(time.time() - t0)
    #     print(i+1, '/', itr, ', time:', naive_time[i])
    # dt = np.mean(naive_time)
    # label.append('Naive')
    # times.append(dt)
    # print(f'Mean execution time: {dt:.2f} seconds\n')
    # mf.plotter(heatmap_naive, f'Naive ({dt:.2f} s)', showImg, saveImg, 'naive.pdf')
    
    print('Numpy Mandelbrot implementation...')
    numpy_time = []
    for i in range(itr):
        t0 = time.time()
        heatmap_numpy = mf.mandelbrot_numpy(c, T, I)
        numpy_time.append(time.time() - t0)
        print(i+1, '/', itr, ', time:', numpy_time[i])
    dt = np.mean(numpy_time)
    label.append('Numpy')
    times.append(dt)
    print(f'Mean execution time: {dt:.2f} seconds\n')
    mf.plotter(heatmap_numpy, f'Numpy ({dt:.2f} s)', showImg, saveImg, 'numpy.pdf')
    
    print('Numba optimized Mandelbrot implementation...')
    numba_time = []
    for i in range(itr):
        t0 = time.time()
        heatmap_numba = mf.mandelbrot_numba(c, T, I)
        numba_time.append(time.time() - t0)
        print(i+1, '/', itr, ', time:', numba_time[i])
    dt = np.mean(numba_time)
    label.append('Numba')
    times.append(dt)
    print(f'Mean execution time: {dt:.2f} seconds\n')
    mf.plotter(heatmap_numba, f'Numba optimized ({dt:.2f} s)', showImg, saveImg, 'numba.pdf')
    
    print('Parallel multi-processing Mandelbrot implementation...')
    parallel_time = []
    for i in range(itr):
        t0 = time.time()
        heatmap_parallel = mf.mandelbrot_parallel(c, os.cpu_count(), alpha, alpha)
        parallel_time.append(time.time() - t0)
        print(i+1, '/', itr, ', time:', parallel_time[i])
    dt = np.mean(parallel_time)
    label.append('Parallel')
    times.append(dt)
    print(f'Mean execution time: {dt:.2f} seconds\n')
    mf.plotter(heatmap_parallel, f'Parallel optimized ({dt:.2f} s)', showImg, saveImg, 'parallel.pdf')

    print('Dask Mandelbrot implementation...')
    dask_time = []
    for i in range(itr):
        t0 = time.time()
        heatmap_dask = mf.mandelbrot_dask(c, T, I)
        dask_time.append(time.time() - t0)
        print(i+1, '/', itr, ', time:', dask_time[i])
    dt = np.mean(dask_time)
    label.append('Dask')
    times.append(dt)
    print(f'Mean execution time: {dt:.2f} seconds\n')
    mf.plotter(heatmap_dask, f'Dask ({dt:.2f} s)', showImg, saveImg, 'dask.pdf')

    print('GPU Mandelbrot implementation...')
    gpu_time = []
    for i in range(itr):
        t0 = time.time()
        heatmap_gpu = mf.mandelbrot_gpu(c, T, I, pl_id=2, dev_id=0, group_size=(32, 32))
        gpu_time.append(time.time() - t0)
        print(i+1, '/', itr, ', time:', gpu_time[i])
    dt = np.mean(gpu_time)
    label.append('GPU')
    times.append(dt)
    print(f'Mean execution time: {dt:.2f} seconds\n')
    mf.plotter(heatmap_gpu, f'GPU ({dt:.2f} s)', showImg, saveImg, 'gpu.pdf')

    mf.comparison_plot(label, times, showImg, saveImg)
    print('Finished')