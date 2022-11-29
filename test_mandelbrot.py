"""
Pytest file for mandelbrot_functions.py to check functionality. This file 
contains two classes of tests, one for create_mesh, and one for all my
Mandelbrot implementations.

Execution time on authors laptop: < 2 min
"""
import pytest
import numpy as np
import os
import mandelbrot_functions as mf

class TestCreateMesh:
    """
    Class to test function create_mesh
    """

    def test_mesh_shape(self):
        """
        Function to test shape
        """
        points = 64
        c_mesh = mf.create_mesh(points)
        assert c_mesh.shape == (points, points)

    def test_mesh_dtype(self):
        """
        Function to test data type
        """
        points = 64
        c_mesh32 = mf.create_mesh(points, data_type=np.float32)
        c_mesh64 = mf.create_mesh(points, data_type=np.float64)
        assert c_mesh32.dtype == 'complex64'
        assert c_mesh64.dtype == 'complex128'

    def test_mesh_values(self):
        """
        Function to test output values

        Expected values in a 4x4 mesh is compared with actual output values.
        """
        c_mesh = mf.create_mesh(4, data_type=np.float32)
        c_desired = np.array([[-2.+1.5j, -1.+1.5j, 0.+1.5j, 1.+1.5j],
                              [-2.+0.5j, -1.+0.5j, 0.+0.5j, 1.+0.5j],
                              [-2.-0.5j, -1.-0.5j, 0.-0.5j, 1.-0.5j],
                              [-2.-1.5j, -1.-1.5j, 0.-1.5j, 1.-1.5j]],
                              dtype=np.complex64)
        np.testing.assert_array_equal(c_desired, c_mesh)

class TestImplementationOutputs:
    """
    Class to test implemented Mandelbrot functions
    """

    def test_output_shape(self):
        """
        Function to test output shapes of each function
        """
        points = 64
        c = mf.create_mesh(points)

        naive = mf.mandelbrot_naive(c)
        numpy = mf.mandelbrot_numpy(c)
        numba = mf.mandelbrot_numba(c)
        parallel = mf.mandelbrot_parallel(c, os.cpu_count(), 8, 8)
        dask = mf.mandelbrot_dask(c)
        gpu = mf.mandelbrot_gpu(c)

        assert naive.shape == (points, points)
        assert numpy.shape == (points, points)
        assert numba.shape == (points, points)
        assert parallel.shape == (points, points)
        assert dask.shape == (points, points)
        assert gpu.shape == (points, points)

    def test_output_dtype(self):
        """
        Function to test output data type of implemented functions

        Implementations such as naive, numpy, and dask have configurable data
        type in my case, and here the output is testet.
        """
        points = 4
        c_mesh32 = mf.create_mesh(points, np.float32)
        c_mesh64 = mf.create_mesh(points, np.float64)

        naive32 = mf.mandelbrot_naive(c_mesh32, data_type=np.float32)
        naive64 = mf.mandelbrot_naive(c_mesh64, data_type=np.float64)
        numpy32 = mf.mandelbrot_numpy(c_mesh32, data_type=np.float32)
        numpy64 = mf.mandelbrot_numpy(c_mesh64, data_type=np.float64)
        dask32 = mf.mandelbrot_dask(c_mesh32, data_type=np.float32)
        dask64 = mf.mandelbrot_dask(c_mesh64, data_type=np.float64)

        assert naive32.dtype == 'float32'
        assert naive64.dtype == 'float64'
        assert numpy32.dtype == 'float32'
        assert numpy64.dtype == 'float64'  
        assert dask32.dtype == 'float32'
        assert dask64.dtype == 'float64'

    def test_output_values(self):
        """
        Function to test output values of implemented functions

        Expected values in a 4x4 mesh is compared with actual output values.
        """
        points = 4
        c = mf.create_mesh(points, np.float32)

        naive = mf.mandelbrot_naive(c)
        numpy = mf.mandelbrot_numpy(c)
        numba = mf.mandelbrot_numba(c)
        parallel = mf.mandelbrot_parallel(c, os.cpu_count(), 2, 2)
        dask = mf.mandelbrot_dask(c)
        gpu = mf.mandelbrot_gpu(c)

        truth = np.array([[0, 0.04, 0.04, 0.04],
                          [0, 0.16, 1, 0.04],
                          [0, 0.16, 1, 0.04],
                          [0, 0.04, 0.04, 0.04]])

        np.testing.assert_array_almost_equal(naive, truth)
        np.testing.assert_array_almost_equal(numpy, truth)
        np.testing.assert_array_almost_equal(numba, truth)
        np.testing.assert_array_almost_equal(parallel, truth)
        np.testing.assert_array_almost_equal(dask, truth)
        np.testing.assert_array_almost_equal(gpu, truth)

#retcode = pytest.main()