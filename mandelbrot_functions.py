"""
Mandelbrot set functions file. This file contains different implementations to 
calculate the mandelbrot set.
"""
import pyopencl as cl
import os
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
from multiprocessing import Pool
from dask.distributed import Client
from dask import array as da

def create_mesh(points=64, data_type=np.float32):
    """
    Generate mesh in complex plane.
    
    Given the same amount of points is desired for both real and complex 
    values, a plane within -2 < Re < 1, -1.5 < Im < 1.5 is generated. 
    Resolution is determined by amount of points.
    
    Parameters
    ----------
    points : int
        Desired number of real and imaginary points
    data_type : type
        Data type for Re and Im points
        
    Returns
    -------
    mesh : numpy.ndarray
        Mesh in complex plane
    """
    Re = np.array([np.linspace(-2, 1, points),] * points, dtype=data_type)
    Im = np.array([np.linspace(1.5, -1.5, points),] * points, dtype=data_type).transpose()

    mesh = Re + Im * 1j
    
    return mesh

def plotter(c: np.ndarray, title=None, show=True, save=False, fname='plot.pdf'):
    """
    Plot heatmap for calculated Mandelbrot set.
    
    This function will work for the entire Mandelbrot set, spanning fron
    -2 < Re < 1, -1.5 < Im < 1.5.
    
    Parameters
    ----------
    c : numpy.ndarray
        Input mesh to plot
    title : str
        Plot title, e.g. 'Naive (123 s)'
    show : bool
        Show the plot if True
    save : bool
        Save the plot if True
    fname : str
        Filename to save image, in local folder Plots
    """
    plt.clf()
    plt.imshow(c, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(title)
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")  
    if save:
        fname = os.path.join('Plots/', fname)
        plt.savefig(fname, dpi=(200))
    if show:
        plt.show() 
        
def comparison_plot(labels, times, show=True, save=False, fname='performance_comparison.pdf'):
    """
    Plot comparison of given labelled execution times.
    
    This function will make a plot for easy comparison of execution times. A 
    list of labels must follow, that is in the same order as execution times.
    
    Parameters
    ----------
    labels : list
        List of labels
    times : list
        List of executions times
    show : bool
        Show the plot if True
    save : bool
        Save the plot if True
    fname : str
        Filename to save image, in local folder Plots
    """
    plt.clf()
    n = [i+1 for i,v in enumerate(times)]
    plt.bar(n,times,tick_label = labels)
    plt.xlabel('Implementation')
    plt.ylabel('Execution time')
    plt.title('Performance comparison of implementations')
    for i in range(len(times)):
        plt.annotate(str(round(times[i],2)), xy=(n[i],times[i]), ha='center', va='bottom')
    if save:
        fname = os.path.join('Plots/', fname)
        plt.savefig(fname, dpi=(200))
    if show:
        plt.show() 
        
def mandelbrot_naive(c, T=2, I=25, data_type=np.float32):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is a naive Python implementation using nested for-loops.
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane
    T : int
        Output values threshold
    I : int
        Max iterations         
    data_type : type
        Output data type
        
    Returns
    -------
    output : numpy.ndarray
        Naive Mandelbrot set
    """
    n = np.zeros_like(c, dtype=np.int16)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = c[i,j]
            while abs(z) <= T and n[i,j] < I:
                z = z*z + c[i,j]
                n[i,j] += 1
    output = data_type(n/I)
    return output

def mandelbrot_numpy(c, T=2, I=25, data_type=np.float32):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is a vectorized implementation using Numpy arrays.
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane
    T : int
        Output values threshold
    I : int
        Max iterations         
    data_type : type
        Output data type
        
    Returns
    -------
    output : numpy.ndarray
        Numpy Mandelbrot set
    """
    z = np.zeros(c.shape, dtype=data_type)
    iota = np.zeros(c.shape,dtype=np.int16)
    idx = np.full(c.shape,True)      
    for i in range(I):
        z = z*z + c
        idx = np.abs(z) <= T
        if ~idx.any():
            break
        iota[idx] = i+1
        z[~idx] = T
        output = data_type(iota/I)
    return output

@jit(nopython=True)
def mandelbrot_numba(c, T=2, I=25):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is a naive Python implementation using nested for-loops, but
    optimized with Numba.
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane
    T : int
        Output values threshold
    I : int
        Max iterations
    data_type : type
        Output data type
        
    Returns
    -------
    output : numpy.ndarray
        Numba Mandelbrot set
    """
    n = np.zeros_like(c, dtype=numba.int64)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = c[i,j]
            while abs(z) <= T and n[i,j] < I:
                z = z*z + c[i,j]
                n[i,j] += 1
    output = n/I
    return output

def mandelbrot_parallel(c, P, L, N):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is running the Numpy function in parallel using the
    multiprocessing library. Amount of workers are assigned, and the
    input mesh is divided horizontally.
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane   
    P : int
        Number of workers
    L : int
        Number of blocks do divide c into (horizontally split)
    N : int
        Number of rows inside each block
        
    Returns
    -------
    output : numpy.ndarray
        Parallel Mandelbrot set
    """
    chunk = [c[N*b:N*b+N] for b in range(L)]
    with Pool(processes=P) as pool:
        result = pool.map_async(mandelbrot_numpy, chunk)
        pool.close()
        pool.join()
        output = np.vstack([row for row in result.get()])
    return output

def mandelbrot_dask(c, T=2, I=25, chunksize='auto', data_type=np.float32):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is a vectorized implementation using Dask arrays.
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane
    T : int
        Output values threshold
    I : int
        Max iterations
    chunks : int or list
        Size of chunks in Dask array
    data_type : type
        Output data type
        
    Returns
    -------
    output : numpy.ndarray
        Dask Mandelbrot set
    """
    client = Client()
    c_dask = da.from_array(c, chunks=chunksize)
    M_dask = da.map_blocks(mandelbrot_numpy, c_dask, T, I, data_type, dtype=data_type)
    output = M_dask.compute()
    client.close()
    return output

def introspection():
    """
    Run to check available OpenCL platforms and devices
    """
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    # Print each platform on this computer
    for platform in cl.get_platforms():
        print('=' * 60)
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        # Print each device per-platform
        for device in platform.get_devices():
            print('    ' + '-' * 56)
            print('    Device - Name:  ' + device.name)
            print('    Device - Type:  ' + cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024.0))
            print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024.0))
            print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
            print('    Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size/1048576.0))
            print('    Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
    print('\n')

def mandelbrot_gpu(c, T=2, I=25, pl_id=2, dev_id=0, data_type=np.float64, group_size=None):
    """
    Calculate Mandelbrot set given a mesh in the complex plane.
    
    This function is performing computations on the GPU by using the
    pyopencl library. The C++ kernel is coded in-line this function. 
    
    Parameters
    ----------
    c : numpy.ndarray
        Mesh in the complex plane
    T : int
        Output values threshold
    I : int
        Max iterations
    pl_id : int
        Platform ID
    dev_id : int
        Device ID
        
    Returns
    -------
    output : numpy.ndarray
        GPU Mandelbrot set
    """
    ## Initialize PyOpenCL runtime
    # Create context (environment, devices, platform)
    platform = cl.get_platforms()
    device = [platform[pl_id].get_devices()[dev_id]]
    ctx = cl.Context(device)
    # Command queue for operations
    queue = cl.CommandQueue(ctx)
    # Prepare output variable
    result_host = np.empty(c.shape).astype(data_type)
    
    ## Kernel execution steps
    # Allocate device memory and copy intpu to device
    mf = cl.mem_flags
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c.astype(np.complex128))
    y_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_host.nbytes)
    # Define kernel (in-line)
    defines = f"""
    #define PYOPENCL_DEFINE_CDOUBLE
    #include <pyopencl-complex.h>
    #define THRESHOLD {np.float64(T)}
    #define MAX_ITER {np.float64(I)}
    """
    
    kernel_source = """
    __kernel void mandelbrot(
        __global const cdouble_t *x_g,
        __global double *y_g
        )
    {
         __private const int gidx = get_global_id(0);
         __private const int gidy = get_global_id(1);
         __private const int gsx = get_global_size(0);
         
         __private const cdouble_t c = x_g[gidx + gsx * gidy];
         __private cdouble_t z = c;
         __private double n = 0;
         
         while ((cdouble_abs(z) <= THRESHOLD) && (n < MAX_ITER)){
                 z = cdouble_add(cdouble_mul(z, z), c);
                 n = n + 1;
         }
         y_g[gidx + gsx * gidy] = n/MAX_ITER;
     }
    """
    # Build kernel
    prg = cl.Program(ctx, defines + kernel_source).build()
    mandelbrot = prg.mandelbrot
    # Launch kernel
    mandelbrot(queue, c.shape, group_size, x_g, y_g)
    # Copy result back from device to host
    cl.enqueue_copy(queue, result_host, y_g)

    return result_host