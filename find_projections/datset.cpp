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
  training_rows = NULL;
}

double Datset::get_default_value() {
  double value = 0;
  if(is_classifier == true) {
    std::vector<int> class_dist = std::vector<int>(num_classes);
    for(int i=0; i<rows; i++) {
      int value = this->ds_output_ref(i);
      class_dist[value]++;
    }
    int argmax = 0;
    int max = 0;
    for(unsigned int i=0; i<class_dist.size(); i++) {
      if(class_dist[i] > max) {
        max = class_dist[i];
        argmax = i;
      }
    }
    value = argmax;
  }
  else {
    double truemean = 0.0;
    for(int i=0; i<rows; i++) {
      double val = this->ds_output_ref(i);
      truemean += val;
    }
    truemean /= rows;
    value = truemean;
  }

  return value;
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

void Datset::set_training_rows(PyObject *object) {
  if(this->training_rows) {
      delete this->training_rows;
  }

  PyArrayObject *array = reinterpret_cast<PyArrayObject *>(object);
  int rows = PyArray_DIM(array, 0);
  this->training_rows = new std::vector<int>(rows);

  double *iter = reinterpret_cast< double * >( PyArray_GETPTR1(array, 0) );
  for (int i = 0; i < rows; ++i) {
    int val = (int)iter[i];
    (*training_rows)[i] = val;
  }
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
  if(training_rows)
    delete training_rows;
  training_rows = NULL;
}
