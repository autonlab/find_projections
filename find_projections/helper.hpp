/*
   File:        helper.hpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 26 16:23:48 EST 2016
   Description: 
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef HELPER_H_
#define HELPER_H_

#include <vector>
#include "datset.hpp"

#define MIN_DOUBLE 1E-6

class indices_array {
private:
  std::vector<std::vector<int> *> ia;
public:
  std::vector<int> &get_indices(int i) {
    return *(ia[i]);
  }
  ~indices_array() {
    for(unsigned int i=0; i<ia.size(); i++)
      delete ia[i];
  }
  int size() {
    return ia.size();
  }
  indices_array(int rows) : ia(rows) {
  }
  void set_indices(int i, std::vector<int> *ptr) {
    ia[i] = ptr;
  }
};

class Helper {
public:
  static int minimum_of_3(double a, double b, double c) {
    return (a<=b) ? (a<=c ? 0 : 2) : (b<=c ? 1 : 2);
  }

  static int maximum_of_3(double a, double b, double c) {
    return (a>=b) ? (a>=c ? 0 : 2) : (b>=c ? 1 : 2);
  }

  static int max(int a, int b) {
    return (a > b)?a:b;
  }

  /*
   * Sort all dataset attributes except for label.
   * Store indices of sorted values
   */
  static indices_array *mk_indices_array_sorted_values(Datset &ds, std::vector<int> &train_rows);

  /*
   * Retrieve the start-end optimal range from the binary tree built for attribute 'att'.
   * Also retrieves the number of positives and negatives in the box
   */
  static void sort_indices_based_on_values(Datset &ds, int att, std::vector<int> &iv);

  static std::vector<int> *get_vector_subset(std::vector<int> &v, int start, int end);

  static std::vector<int> *intersection(std::vector<int> &v1, std::vector<int> &v2);
};

#endif
