/*
   File:        projection.cpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Represents 2-d subset of data
   Copyright (c) 2016 Carnegie Mellon University
*/

#include "projection.hpp"
#include "helper.hpp"
#include "discrete_projection.hpp"
#include "numeric_projection.hpp"
#include "discrete_binary_tree.hpp"

#include <iostream>

using namespace std;

#include <typeinfo>
#include <string>

static bool point_within_line(double start, double end, Datset &ds, int row, int att) {
  double value = ds.ds_real_ref(row, att);
  return (value >= start && value <= end);
}

/*
 * Returns true if query point lies inside projection
 */
bool projection::point_lies_in_projection(Datset &ds, int row) {
  int att1 = this->att1;
  int att2 = this->att2;

  return (point_within_line(att1_start, att1_end, ds, row, att1) &&
          point_within_line(att2_start, att2_end, ds, row, att2));
}

/*
 * Constructor
 */
projection::projection(int att1, int att2, double att1_start, double att1_end, double att2_start, double att2_end) : 
  att1(att1), att2(att2), att1_start(att1_start), att1_end(att1_end), att2_start(att2_start), att2_end(att2_end) {
  this->indices = NULL;
}

projection::projection() {
  this->indices = NULL;
}

/*
 * Retrieve the start-end optimal range from the binary tree built for attribute 'att'.
 * Also retrieves the number of positives and negatives in the box
 */
static void get_optimal_xrange(std::vector<int> &iv, double *start, double *end, btree_node *node, Datset &ds,  std::vector<int> &train_rows, int att) {
  int opt_first = node->get_node_optimal_start();
  int opt_last = node->get_node_optimal_end();

  *start = ds.ds_real_ref(train_rows[iv[opt_first]], att);
  *end = ds.ds_real_ref(train_rows[iv[opt_last]], att);
}

/*
 * Creates projection from -
 * att1's row subset (start-end) of ivatt1
 * att2's row subset selected in binary tree 'node' of ivatt2
 * node - Binary tree containing optimal range of att2
 */
projection *projection::mk_projection_from_tree(btree_node *node, Datset &ds, std::vector<int> &train_rows, 
						std::vector<int> &ivatt1, int start, int end,
						std::vector<int> &ivatt2, int att1, int att2) {
  double start2, end2;
  
  get_optimal_xrange(ivatt2, &start2, &end2, node, ds, train_rows, att2);

  double start_value = ds.ds_real_ref(train_rows[ivatt1[start]], att1);
  double end_value = ds.ds_real_ref(train_rows[ivatt1[end]], att1);

  projection *pr = NULL;
  string name = node->get_name();
  if(name == "discrete_binary_tree")
    pr = new discrete_projection(att1, att2, start_value, end_value, start2, end2);
  else
    pr = new numeric_projection(att1, att2, start_value, end_value, start2, end2);

  return pr;
}

/*
 * Destructor
 */
projection::~projection() {
  if(this->indices)
    delete this->indices;
  this->indices = NULL;
}

void projection::copy_projection(projection *pr) {
  pr->att1 = this->att1;
  pr->att2 = this->att2;
  pr->att1_start = this->att1_start;
  pr->att1_end = this->att1_end;
  pr->att2_start = this->att2_start;
  pr->att2_end = this->att2_end;
  pr->indices = NULL;
  if(this->indices)
    pr->indices = new std::vector<int>(*(this->indices));
}

static bool do_lines_overlap(double start1, double end1, double start2, double end2) {
  return !(start2 > end1 || end2 < start1);
}
 
/*
 * Returns true if projections are for same pair of attributes and have overlapping boxes
 */
bool projection::do_projections_overlap(projection *pr2) {
  if(!pr2)
    return false;

  if(this->att1 == pr2->att1 &&
     this->att2 == pr2->att2) {
    bool overlap = do_lines_overlap(this->att1_start, this->att1_end, pr2->att1_start, pr2->att1_end) &&
      do_lines_overlap(this->att2_start, this->att2_end, pr2->att2_start, pr2->att2_end);
    return overlap;
  }
  else
    return false;
}

/*
 * Binary search for index
 * Attribute can contain duplicates 
 * For lower end, the earliest index corresponding to the value should be returned.
 * For upper end, the latest index corresponding to the value should be returned.
 */
static int find_index(std::vector<int> &iv, Datset &ds, std::vector<int> &train_rows, double value, int att, bool lower) {
  int lb = 0, ub = iv.size()-1;
  int size = ub;

  while(lb < ub) {
    int M = (lb + ub)/2;
    int row = train_rows[iv[M]];
    double mval = ds.ds_real_ref(row, att);

    if(mval == value) { // Match
      if(!lower) { // Finding upper bound of the range
        if(M+1 > size)
          return M;
        if(ds.ds_real_ref(train_rows[iv[M+1]], att) > value)
          return M;
        else
          lb = M+1;
      }
      else { // Finding lower bound of the range
        if(M == 0)
          return M;
        if(ds.ds_real_ref(train_rows[iv[M-1]], att) < value)
          return M;
        else 
          ub = M-1;
      }
      continue;
    }

    if(mval > value)
      ub = M-1;
    else
      lb = M+1;
  }

  if(lb > ub)
    lb = ub;
  if(lb < 0)
    lb = 0;

  return lb;
}

void projection::mk_projection_indices(Datset &ds, std::vector<int> &train_rows, indices_array &ia) {
  if(this->indices)
    return;
  
  int f1att = att1; /* Y-axis */
  int f2att = att2; /* X-axis */

  std::vector<int> &ivatt1 = ia.get_indices(f1att);
  std::vector<int> &ivatt2 = ia.get_indices(f2att);

  int start_row1 = find_index(ivatt1, ds, train_rows, att1_start, att1, true);
  int end_row1 = find_index(ivatt1, ds, train_rows, att1_end, att1, false);
  std::vector<int> *iv1 = Helper::get_vector_subset(ivatt1, start_row1, end_row1+1);

  int start_row2 = find_index(ivatt2, ds, train_rows, att2_start, att2, true);
  int end_row2 = find_index(ivatt2, ds, train_rows, att2_end, att2, false);
  std::vector<int> *iv2 = Helper::get_vector_subset(ivatt2, start_row2, end_row2+1);

  std::vector<int> *indices = Helper::intersection(*iv1, *iv2);

  delete iv1;
  delete iv2;

  this->indices = indices;
}
