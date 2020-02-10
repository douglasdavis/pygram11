// MIT License

// Copyright (c) 2019 Douglas Davis

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Local
#include "_helpers.hpp"

// Python
#include <Python.h>

// NumPy
#include <numpy/arrayobject.h>

// OpenMP
#include <omp.h>

// C++ STL
#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" {

static PyObject* f1dw(PyObject* self, PyObject* args);
static PyObject* v1dw(PyObject* self, PyObject* args);
static PyObject* omp_gmt(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {{"_f1dw", f1dw, METH_VARARGS, ""},
                                       {"_v1dw", v1dw, METH_VARARGS, ""},
                                       {"_omp_get_max_threads", omp_gmt, METH_NOARGS, ""},
                                       {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "_CPP",
                                       "pygram11's C API based 1D backend",
                                       -1,
                                       module_methods,
                                       NULL,
                                       NULL,
                                       0,
                                       NULL};

__attribute__((visibility("default"))) PyObject* PyInit__CPP(void);
__attribute__((visibility("default"))) PyObject* PyInit__CPP(void) {
  PyObject* m = PyModule_Create(&moduledef);
  import_array();
  return m;
}

}  // extern "C"

enum class Status {
  SUCCESS = 0,
  ERROR = 1,
};

template <typename T1, typename T2>
inline void fixed_serial_fill_include_flow(const T1* x, const T2* w, T2* counts, T2* vars,
                                           long nx, std::size_t nbins, double xmin,
                                           double xmax, double norm) {
  for (long i = 0; i < nx; ++i) {
    auto bin = pygram11::helpers::get_bin(x[i], nbins, xmin, xmax, norm);
    counts[bin] += w[i];
    vars[bin] += w[i] * w[i];
  }
  return;
}

template <typename T1, typename T2>
inline void fixed_serial_fill_exclude_flow(const T1* x, const T2* w, T2* counts, T2* vars,
                                           long nx, std::size_t nbins, double xmin,
                                           double xmax, double norm) {
  for (long i = 0; i < nx; ++i) {
    if (x[i] < xmin || x[i] >= xmax) {
      continue;
    }
    else {
      auto bin = pygram11::helpers::get_bin(x[i], nbins, xmin, norm);
      counts[bin] += w[i];
      vars[bin] += w[i] * w[i];
    }
  }
  return;
}

template <typename T1, typename T2, typename T3>
inline void var_serial_fill_include_flow(const T1* x, const T2* w, T2* counts, T2* vars,
                                         long nx, const std::vector<T3>& edges) {
  std::size_t nbins = static_cast<int>(edges.size()) - 1;
  for (long i = 0; i < nx; ++i) {
    auto bin = pygram11::helpers::get_bin(x[i], nbins, edges);
    counts[bin] += w[i];
    vars[bin] += w[i] * w[i];
  }
}

template <typename T1, typename T2, typename T3>
inline void var_serial_fill_exclude_flow(const T1* x, const T2* w, T2* counts, T2* vars,
                                         long nx, const std::vector<T3>& edges) {
  for (long i = 0; i < nx; ++i) {
    if (x[i] < edges.front() || x[i] >= edges.back()) {
      continue;
    }
    else {
      auto bin = pygram11::helpers::get_bin(x[i], edges);
      counts[bin] += w[i];
      vars[bin] += w[i] * w[i];
    }
  }
}

template <typename T1, typename T2>
inline void fixed_fill_include_flow(const T1* x, const T2* w, T2* counts, T2* vars, long nx,
                                    std::size_t nbins, double xmin, double xmax,
                                    double norm) {
  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (long i = 0; i < nx; ++i) {
      bin = pygram11::helpers::get_bin(x[i], nbins, xmin, xmax, norm);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
  Py_END_ALLOW_THREADS;
}

template <typename T1, typename T2>
inline void fixed_fill_exclude_flow(const T1* x, const T2* w, T2* counts, T2* vars, long nx,
                                    std::size_t nbins, double xmin, double xmax,
                                    double norm) {
  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (long i = 0; i < nx; ++i) {
      if (x[i] < xmin || x[i] >= xmax) {
        continue;
      }
      else {
        bin = pygram11::helpers::get_bin(x[i], nbins, xmin, norm);
        weight = w[i];
        counts_ot[bin] += weight;
        vars_ot[bin] += weight * weight;
      }
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
  Py_END_ALLOW_THREADS;
}

template <typename T1, typename T2, typename T3>
inline void var_fill_include_flow(const T1* x, const T2* w, T2* counts, T2* vars, long nx,
                                  const std::vector<T3>& edges) {
  std::size_t nbins = static_cast<int>(edges.size()) - 1;
  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (long i = 0; i < nx; ++i) {
      bin = pygram11::helpers::get_bin(x[i], nbins, edges);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
  Py_END_ALLOW_THREADS;
}

template <typename T1, typename T2, typename T3>
inline void var_fill_exclude_flow(const T1* x, const T2* w, T2* counts, T2* vars, long nx,
                                  const std::vector<T3>& edges) {
  std::size_t nbins = static_cast<int>(edges.size()) - 1;
  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (long i = 0; i < nx; ++i) {
      if (x[i] < edges.front() || x[i] >= edges.back()) {
        continue;
      }
      else {
        bin = pygram11::helpers::get_bin(x[i], edges);
        weight = w[i];
        counts_ot[bin] += weight;
        vars_ot[bin] += weight * weight;
      }
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
  Py_END_ALLOW_THREADS;
}

#define FILL_CALL_FIXED(IS1, IS2, T1, T2, suffix)                                         \
  do {                                                                                    \
    if (x_is_##IS1 && w_is_##IS2) {                                                       \
      if (nx < 5000) {                                                                    \
        fixed_serial_fill_##suffix<T1, T2>(                                               \
            (const T1*)PyArray_DATA(x), (const T2*)PyArray_DATA(w), (T2*)PyArray_DATA(c), \
            (T2*)PyArray_DATA(v), nx, nbins, xmin, xmax, norm);                           \
      }                                                                                   \
      else {                                                                              \
        fixed_fill_##suffix<T1, T2>((const T1*)PyArray_DATA(x),                           \
                                    (const T2*)PyArray_DATA(w), (T2*)PyArray_DATA(c),     \
                                    (T2*)PyArray_DATA(v), nx, nbins, xmin, xmax, norm);   \
      }                                                                                   \
      return Status::SUCCESS;                                                             \
    }                                                                                     \
  } while (0)

#define FILL_CALL_VAR(IS1, IS2, T1, T2, suffix)                                            \
  do {                                                                                     \
    if (x_is_##IS1 && w_is_##IS2) {                                                        \
      if (nx < 10000) {                                                                    \
        var_serial_fill_##suffix<T1, T2>((const T1*)PyArray_DATA(x),                       \
                                         (const T2*)PyArray_DATA(w), (T2*)PyArray_DATA(c), \
                                         (T2*)PyArray_DATA(v), nx, edges);                 \
      }                                                                                    \
      else {                                                                               \
        var_fill_##suffix<T1, T2>((const T1*)PyArray_DATA(x), (const T2*)PyArray_DATA(w),  \
                                  (T2*)PyArray_DATA(c), (T2*)PyArray_DATA(v), nx, edges);  \
      }                                                                                    \
      return Status::SUCCESS;                                                              \
    }                                                                                      \
  } while (0)

#define CHECK_TYPES                                                                  \
  bool x_is_float64, x_is_float32, x_is_uint32, x_is_int32, x_is_uint64, x_is_int64, \
      w_is_float64, w_is_float32;                                                    \
  do {                                                                               \
    x_is_float64 = PyArray_TYPE(x) == NPY_FLOAT64;                                   \
    x_is_float32 = PyArray_TYPE(x) == NPY_FLOAT32;                                   \
    x_is_uint32 = PyArray_TYPE(x) == NPY_UINT32;                                     \
    x_is_int32 = PyArray_TYPE(x) == NPY_INT32;                                       \
    x_is_uint64 = PyArray_TYPE(x) == NPY_UINT64;                                     \
    x_is_int64 = PyArray_TYPE(x) == NPY_INT64;                                       \
    w_is_float64 = PyArray_TYPE(w) == NPY_FLOAT64;                                   \
    w_is_float32 = PyArray_TYPE(w) == NPY_FLOAT32;                                   \
  } while (0)

static Status fill_f1dw_include_flow(PyArrayObject* x, PyArrayObject* w, PyArrayObject* c,
                                     PyArrayObject* v, long nx, std::size_t nbins,
                                     double xmin, double xmax) {
  CHECK_TYPES;
  double norm = 1.0 / (xmax - xmin);
  FILL_CALL_FIXED(float32, float32, float, float, include_flow);
  FILL_CALL_FIXED(float64, float32, double, float, include_flow);
  FILL_CALL_FIXED(float32, float64, float, double, include_flow);
  FILL_CALL_FIXED(float64, float64, double, double, include_flow);
  FILL_CALL_FIXED(uint32, float32, unsigned int, float, include_flow);
  FILL_CALL_FIXED(int32, float32, int, float, include_flow);
  FILL_CALL_FIXED(uint32, float64, unsigned int, double, include_flow);
  FILL_CALL_FIXED(int32, float64, int, double, include_flow);
  FILL_CALL_FIXED(uint64, float32, unsigned long, float, include_flow);
  FILL_CALL_FIXED(int64, float32, long, float, include_flow);
  FILL_CALL_FIXED(uint64, float64, unsigned long, double, include_flow);
  FILL_CALL_FIXED(int64, float64, long, double, include_flow);
  return Status::ERROR;
}

static Status fill_f1dw_exclude_flow(PyArrayObject* x, PyArrayObject* w, PyArrayObject* c,
                                     PyArrayObject* v, long nx, std::size_t nbins,
                                     double xmin, double xmax) {
  CHECK_TYPES;
  double norm = 1.0 / (xmax - xmin);
  FILL_CALL_FIXED(float32, float32, float, float, exclude_flow);
  FILL_CALL_FIXED(float64, float32, double, float, exclude_flow);
  FILL_CALL_FIXED(float32, float64, float, double, exclude_flow);
  FILL_CALL_FIXED(float64, float64, double, double, exclude_flow);
  FILL_CALL_FIXED(uint32, float32, unsigned int, float, exclude_flow);
  FILL_CALL_FIXED(int32, float32, int, float, exclude_flow);
  FILL_CALL_FIXED(uint32, float64, unsigned int, double, exclude_flow);
  FILL_CALL_FIXED(int32, float64, int, double, exclude_flow);
  FILL_CALL_FIXED(uint64, float32, unsigned long, float, exclude_flow);
  FILL_CALL_FIXED(int64, float32, long, float, exclude_flow);
  FILL_CALL_FIXED(uint64, float64, unsigned long, double, exclude_flow);
  FILL_CALL_FIXED(int64, float64, long, double, exclude_flow);
  return Status::ERROR;
}

static Status fill_v1dw_include_flow(PyArrayObject* x, PyArrayObject* w, PyArrayObject* c,
                                     PyArrayObject* v, long nx, PyArrayObject* ed,
                                     int nedges) {
  const double* edges_arr = (const double*)PyArray_DATA(ed);
  std::vector<double> edges(edges_arr, edges_arr + nedges);
  CHECK_TYPES;
  FILL_CALL_VAR(float32, float32, float, float, include_flow);
  FILL_CALL_VAR(float64, float32, double, float, include_flow);
  FILL_CALL_VAR(float32, float64, float, double, include_flow);
  FILL_CALL_VAR(float64, float64, double, double, include_flow);
  FILL_CALL_VAR(uint32, float32, unsigned int, float, include_flow);
  FILL_CALL_VAR(int32, float32, int, float, include_flow);
  FILL_CALL_VAR(uint32, float64, unsigned int, double, include_flow);
  FILL_CALL_VAR(int32, float64, int, double, include_flow);
  FILL_CALL_VAR(uint64, float32, unsigned long, float, include_flow);
  FILL_CALL_VAR(int64, float32, long, float, include_flow);
  FILL_CALL_VAR(uint64, float64, unsigned long, double, include_flow);
  FILL_CALL_VAR(int64, float64, long, double, include_flow);
  return Status::ERROR;
}

static Status fill_v1dw_exclude_flow(PyArrayObject* x, PyArrayObject* w, PyArrayObject* c,
                                     PyArrayObject* v, long nx, PyArrayObject* ed,
                                     int nedges) {
  const double* edges_arr = (const double*)PyArray_DATA(ed);
  std::vector<double> edges(edges_arr, edges_arr + nedges);
  CHECK_TYPES;
  FILL_CALL_VAR(float32, float32, float, float, exclude_flow);
  FILL_CALL_VAR(float64, float32, double, float, exclude_flow);
  FILL_CALL_VAR(float32, float64, float, double, exclude_flow);
  FILL_CALL_VAR(float64, float64, double, double, exclude_flow);
  FILL_CALL_VAR(uint32, float32, unsigned int, float, exclude_flow);
  FILL_CALL_VAR(int32, float32, int, float, exclude_flow);
  FILL_CALL_VAR(uint32, float64, unsigned int, double, exclude_flow);
  FILL_CALL_VAR(int32, float64, int, double, exclude_flow);
  FILL_CALL_VAR(uint64, float32, unsigned long, float, exclude_flow);
  FILL_CALL_VAR(int64, float32, long, float, exclude_flow);
  FILL_CALL_VAR(uint64, float64, unsigned long, double, exclude_flow);
  FILL_CALL_VAR(int64, float64, long, double, exclude_flow);
  return Status::ERROR;
}

static PyObject* f1dw(PyObject* Py_UNUSED(self), PyObject* args) {
  unsigned long nx, nw;
  unsigned long nbins;
  int flow, density, as_err;
  double xmin, xmax;
  PyObject *x_obj, *w_obj, *counts_obj, *vars_obj;
  PyArrayObject *x_array, *w_array, *counts_array, *vars_array;
  npy_intp dims[1];

  if (!PyArg_ParseTuple(args, "OOkddppp", &x_obj, &w_obj, &nbins, &xmin, &xmax, &flow,
                        &density, &as_err)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing function input");
    return NULL;
  }

  x_array = (PyArrayObject*)PyArray_FROM_OF(x_obj, NPY_ARRAY_IN_ARRAY);
  w_array = (PyArrayObject*)PyArray_FROM_OF(w_obj, NPY_ARRAY_IN_ARRAY);

  if (x_array == NULL || w_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Could not read input data or weights as array");
    Py_XDECREF(x_array);
    Py_XDECREF(w_array);
    return NULL;
  }

  nx = (unsigned long)PyArray_DIM(x_array, 0);
  nw = (unsigned long)PyArray_DIM(w_array, 0);
  if (nx != nw) {
    PyErr_SetString(PyExc_ValueError, "data and weights must have equal length");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    return NULL;
  }

  dims[0] = nbins;
  counts_obj = PyArray_ZEROS(1, dims, PyArray_TYPE(w_array), 0);
  vars_obj = PyArray_ZEROS(1, dims, PyArray_TYPE(w_array), 0);

  if (counts_obj == NULL || vars_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Could not build output");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_XDECREF(counts_obj);
    Py_XDECREF(vars_obj);
    return NULL;
  }

  counts_array = (PyArrayObject*)counts_obj;
  vars_array = (PyArrayObject*)vars_obj;

  Status fill_result = Status::ERROR;
  if (flow) {
    fill_result = fill_f1dw_include_flow(x_array, w_array, counts_array, vars_array, nx,
                                         nbins, xmin, xmax);
  }
  else {
    fill_result = fill_f1dw_exclude_flow(x_array, w_array, counts_array, vars_array, nx,
                                         nbins, xmin, xmax);
  }
  if (fill_result != Status::SUCCESS) {
    PyErr_SetString(PyExc_TypeError, "dtype of input arrays unsupported");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_DECREF(counts_obj);
    Py_DECREF(vars_obj);
    return NULL;
  }

  if (density) {
    if (PyArray_TYPE(w_array) == NPY_FLOAT64) {
      pygram11::helpers::densify((double*)PyArray_DATA(counts_array),
                                 (double*)PyArray_DATA(vars_array), nbins, xmin, xmax);
    }
    else if (PyArray_TYPE(w_array) == NPY_FLOAT32) {
      pygram11::helpers::densify((float*)PyArray_DATA(counts_array),
                                 (float*)PyArray_DATA(vars_array), nbins, xmin, xmax);
    }
  }

  if (as_err) {
    if (PyArray_TYPE(vars_array) == NPY_FLOAT64) {
      pygram11::helpers::array_sqrt((double*)PyArray_DATA(vars_array), nbins);
    }
    if (PyArray_TYPE(vars_array) == NPY_FLOAT32) {
      pygram11::helpers::array_sqrt((float*)PyArray_DATA(vars_array), nbins);
    }
  }

  Py_DECREF(x_array);
  Py_DECREF(w_array);

  return Py_BuildValue("OO", counts_obj, vars_obj);
}

static PyObject* v1dw(PyObject* Py_UNUSED(self), PyObject* args) {
  unsigned long nx, nw;
  unsigned long ne;
  int flow, density, as_err;
  PyObject *x_obj, *w_obj, *e_obj, *counts_obj, *vars_obj;
  PyArrayObject *x_array, *w_array, *e_array, *counts_array, *vars_array;
  npy_intp dims[1];

  if (!PyArg_ParseTuple(args, "OOOppp", &x_obj, &w_obj, &e_obj, &flow, &density, &as_err)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing function input");
    return NULL;
  }

  x_array = (PyArrayObject*)PyArray_FROM_OF(x_obj, NPY_ARRAY_IN_ARRAY);
  w_array = (PyArrayObject*)PyArray_FROM_OF(w_obj, NPY_ARRAY_IN_ARRAY);
  e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (x_array == NULL || w_array == NULL || e_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Could not read input data or weights as array");
    Py_XDECREF(x_array);
    Py_XDECREF(w_array);
    Py_XDECREF(e_array);
    return NULL;
  }

  nx = (unsigned long)PyArray_DIM(x_array, 0);
  nw = (unsigned long)PyArray_DIM(w_array, 0);
  ne = (unsigned long)PyArray_DIM(e_array, 0);
  if (nx != nw) {
    PyErr_SetString(PyExc_ValueError, "data and weights must have equal length");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    return NULL;
  }

  dims[0] = ne - 1;
  counts_obj = PyArray_ZEROS(1, dims, PyArray_TYPE(w_array), 0);
  vars_obj = PyArray_ZEROS(1, dims, PyArray_TYPE(w_array), 0);

  if (counts_obj == NULL || vars_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Could not build output");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_DECREF(e_array);
    Py_XDECREF(counts_obj);
    Py_XDECREF(vars_obj);
    return NULL;
  }

  counts_array = (PyArrayObject*)counts_obj;
  vars_array = (PyArrayObject*)vars_obj;

  Status fill_result = Status::ERROR;
  if (flow) {
    fill_result =
        fill_v1dw_include_flow(x_array, w_array, counts_array, vars_array, nx, e_array, ne);
  }
  else {
    fill_result =
        fill_v1dw_exclude_flow(x_array, w_array, counts_array, vars_array, nx, e_array, ne);
  }
  if (fill_result != Status::SUCCESS) {
    PyErr_SetString(PyExc_TypeError, "dtype of input arrays unsupported");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_DECREF(e_array);
    Py_DECREF(counts_obj);
    Py_DECREF(vars_obj);
    return NULL;
  }

  if (density) {
    if (PyArray_TYPE(w_array) == NPY_FLOAT64) {
      pygram11::helpers::densify<double>((double*)PyArray_DATA(counts_array),
                                         (double*)PyArray_DATA(vars_array),
                                         (double*)PyArray_DATA(e_array), ne - 1);
    }
    else if (PyArray_TYPE(w_array) == NPY_FLOAT32) {
      pygram11::helpers::densify<float>((float*)PyArray_DATA(counts_array),
                                        (float*)PyArray_DATA(vars_array),
                                        (double*)PyArray_DATA(e_array), ne - 1);
    }
  }

  if (as_err) {
    if (PyArray_TYPE(vars_array) == NPY_FLOAT64) {
      pygram11::helpers::array_sqrt((double*)PyArray_DATA(vars_array), ne - 1);
    }
    if (PyArray_TYPE(vars_array) == NPY_FLOAT32) {
      pygram11::helpers::array_sqrt((float*)PyArray_DATA(vars_array), ne - 1);
    }
  }

  Py_DECREF(x_array);
  Py_DECREF(w_array);
  Py_DECREF(e_array);

  return Py_BuildValue("OO", counts_obj, vars_obj);
}

static PyObject* omp_gmt(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args)) {
  long nthreads = omp_get_max_threads();
  return PyLong_FromLong(nthreads);
}
