/*
 *    File:        datset.cpp
 *    Author(s):   Saswati Ray
 *    Created:     Wed Aug 30 10:19:52 EDT 2017
 *    Description: 
 *    Copyright (c) 2017 Carnegie Mellon University
 *    */

#include "datset.hpp"

Datset::Datset(PyObject *object) {
  PyArrayObject *array = reinterpret_cast<PyArrayObject *>(object);
  rows = PyArray_DIM(array, 0);
  cols = PyArray_DIM(array, 1);
  num_classes = -1;
 
  double *iter = reinterpret_cast< double * >( PyArray_GETPTR2(array, 0, 0) );

  darray = new matrix<double>(rows, cols);
  
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double val = iter[i*cols + j];
      (*darray)(i, j) = val;
    }
  }

  output_class = NULL;
  output_regress = NULL;
}

double Datset::ds_real_ref(int i, int j) {
  return (*darray)(i, j);
}

double Datset::ds_output_ref(int i) {
  double val = (is_classifier)?(*output_class)[i]:(*output_regress)[i];
  return val;
}

void Datset::fill_datset_output_for_classification(PyObject *object) {
  is_classifier = true;

  if(output_class)
    delete output_class;
  if(output_regress)
    delete output_regress;
  output_class = NULL;
  output_regress = NULL;

  output_class = new std::vector<int>(rows);

  PyArrayObject *array = reinterpret_cast<PyArrayObject *>(object);
  double *iter = reinterpret_cast< double * >( PyArray_GETPTR1(array, 0) );
  for (int i = 0; i < rows; ++i) {
    int val = (int)iter[i];
    (*output_class)[i] = val;
 }

  std::vector<int> v(output_class->size());
  for(unsigned int i =0; i<output_class->size(); i++)
    v[i] = (int)output_class->at(i);

  // populate v with data
  std::sort(v.begin(), v.end());
  int uniqueCount = std::unique(v.begin(), v.end()) - v.begin();
  this->num_classes = uniqueCount;
}

void Datset::fill_datset_output_for_regression(PyObject *object) {
  is_classifier = false;

  if(output_class)
    delete output_class;
  if(output_regress)
    delete output_regress;
  output_class = NULL;
  output_regress = NULL;
  this->num_classes = -1;

  output_regress = new std::vector<double>(rows);
  PyArrayObject *array = reinterpret_cast<PyArrayObject *>(object);
  double *iter = reinterpret_cast< double * >( PyArray_GETPTR1(array, 0) );
  for (int i = 0; i < rows; ++i) {
    double val = iter[i];
    (*output_regress)[i] = val;
  }
}

Datset::~Datset() {
  if(darray)
	delete darray;
  darray = NULL;
  if(output_class)
	delete output_class;
  output_class = NULL;
  if(output_regress)
	delete output_regress;
  output_regress = NULL;
}
