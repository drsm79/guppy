//------------------------------------------//
// Author: 		Joe Jenkinson
// Date: 		5th August 2011
// Description: 	
//		
// a Python extension to calculate an
// N-dimension grid of points which can 
// be evaluated on a GPU.
//------------------------------------------//

#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include "numpy/arrayobject.h"


//------------------------------//
// C function to calculate grid
//------------------------------// 
double* creategrid_c(int dimc, double **dims, int *size)
{
int i=0, j=0;
int idim=0, index;
int variablelimit = 0;
double *output;
double dimlow, dimhigh;
double step;
int dimcount;

//Get size and allocate memory
*size = 1;
for (i=0; i<dimc; i++)
      *size *= dims[i][2];
output = (double*)malloc(*size*sizeof(double)*dimc);
variablelimit=*size;

//Iterate over data
do
      {
      dimlow = dims[idim][0]; //lower integration limit for this dimension
      dimhigh = dims[idim][1]; //upper integration limit for this dimension
      dimcount = dims[idim][2]; //number of points to evaluate between limits for the dimension
      step = (dimhigh-dimlow) * (1.0f/dimcount);
      variablelimit /= dimcount; 
      index = idim; //initial index offset
      do
              {
              for (i=0; i<dimcount; i++)
                      for (j=0; j<variablelimit; j++)
                              {
                              output[index] = dimlow+i*step;
                              index += dimc;
                              }
              } while (index < *size*dimc);
        idim++;
        } while (idim < dimc);

//return pointer to results
return output;
}

//---------------------------------//
// Function callable from Python:
// returns tuple of dimensions for 
// GPU output array
//--------------------------------//

static PyObject *
getoutputshape(PyObject *self, PyObject *args)
{
int idim,n=0;
PyObject *input, *result;
PyArrayObject *input_arr;
double dsize;
long size;

if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;
input_arr = (PyArrayObject *)PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
n = input_arr->dimensions[0];
result = PyTuple_New(n);


for (idim=0; idim < n; idim++)
	{
	dsize =  *(double *)(input_arr->data+ idim*input_arr->strides[0] + 2*input_arr->strides[1]);
	size = (long)dsize;
	PyTuple_SetItem(result, idim, PyInt_FromLong(size));
	}
return result;
}

//------------------------------------------------------//
// Function callable from Python:
// calculates all grid points, returns as flat 1D array
//------------------------------------------------------//

static PyObject * 
creategrid(PyObject *self, PyObject *args)
{
npy_intp dimensions[1];

int i,j, n;
double** input_params;
double* data;
PyObject *input;
PyArrayObject *input_arr, *result;

if (!PyArg_ParseTuple(args, "O", &input))
	return NULL;
input_arr = (PyArrayObject *)PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);

n=input_arr->dimensions[0];

input_params = (double**)malloc(sizeof(double*)*n);
for (i=0; i<n; i++)
	{
	input_params[i] = (double *)malloc(sizeof(double)*3);
	for (j=0; j<3; j++)
		input_params[i][j] = *(double *)(input_arr->data+ i*input_arr->strides[0] + j*input_arr->strides[1]);
	}

int size;
data = creategrid_c(n, input_params, &size);
dimensions[0] = size*n;
result = (PyArrayObject *)PyArray_SimpleNewFromData(1, dimensions, PyArray_DOUBLE, (char*)data);
return PyArray_Return(result);
}

//----------------------//
// Define methods list
//----------------------//

PyMethodDef methods[] = {
    {"creategrid", creategrid, METH_VARARGS, "Returns a flattened 1-dimensional Numpy array of all variable combinations"},
    {"getoutputshape", getoutputshape, METH_VARARGS, "Returns a tuple describing the shape of the output array"},
    {NULL, NULL, 0, NULL}
};

//--------------------------//
// Initialise Python Module
//--------------------------//

PyMODINIT_FUNC
initfunctiongenerator(void)
{
	(void) Py_InitModule("functiongenerator", methods);
	import_array();
}


