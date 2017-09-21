/*
   File:        numeric_binary_tree.cpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Binary tree for finding low variance(low stderror) regions
   Copyright (c) 2016 Carnegie Mellon University
*/

#include "numeric_binary_tree.hpp"
#include "helper.hpp"

/*
 * Constructor
 */
numeric_binary_tree::numeric_binary_tree(int first, int last, double end_value) : btree_node(first, last, end_value) {
  this->mean = this->left_mean = this->right_mean = this->optimal_mean = 0;
  this->leftn = this->rightn = this->total = this->optimaln = 0;
  this->total_sum = this->left_sum = this->right_sum = this->optimal_sum = 1E32;
  this->mode = 0;
}

std::string numeric_binary_tree::get_name() {
  return std::string("numeric_binary_tree");
}

/*
 * Constructor for creating parent node from child nodes
 * right_child may be NULL
 */
numeric_binary_tree::numeric_binary_tree(btree_node *left_child, btree_node *right_child) : btree_node(left_child, right_child) {
  this->mean = this->left_mean = this->right_mean = this->optimal_mean = 0;
  this->leftn = this->rightn = this->total = this->optimaln = 0;
  this->total_sum = this->left_sum = this->right_sum = this->optimal_sum = 1E32;
  this->mode = 0;
}

/*
 * Reset tree counts 
 */
void numeric_binary_tree::reset_node(bool exclude_leaves) {
  if(exclude_leaves && is_leaf())
    return;

  this->total_sum = this->left_sum = this->right_sum = this->optimal_sum = 1E32;
  this->optimal_start = this->first;
  this->optimal_end = this->last;

  this->mean = this->left_mean = this->right_mean = this->optimal_mean = 0;
  this->leftn = this->rightn = this->total = this->optimaln = 0;
}

/*
 * Destructor
 */
numeric_binary_tree::~numeric_binary_tree() {
}

/*
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 */
static double get_aggregate_mean(double mean1, double mean2, int n1, int n2) {
  double delta = mean2 - mean1;
  double newmean = mean1 + (delta * n2)/Helper::max(n1 + n2, 1);
  return newmean;
}

/*
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 */
static double get_aggregate_sum_sq_error(double sum_sq_error1, double sum_sq_error2, double mean1, double mean2, int n1, int n2) {
  if(n1 == 0)
    return sum_sq_error2;
  if(n2 == 0)
    return sum_sq_error1;

  double delta = mean2 - mean1;
  delta *= delta;

  double newsum_sq_error = sum_sq_error1 + sum_sq_error2 + (delta * n1 * n2)/Helper::max(n1 + n2, 1);
  return newsum_sq_error;
}

/* Returns width of 1 standard error */
static double get_confidence_band(double mean, double sum_sq_error, int n) {
  double sdev = sqrt(sum_sq_error/Helper::max(n-1, 1));
  double sterr = sdev/sqrt(Helper::max(n, 1));
  return sterr;
}

/*
 * Returns the path with lowest (tightest) confidence band of 1 std error.
 */
int numeric_binary_tree::get_optimal_path(numeric_binary_tree *left, numeric_binary_tree *right) {
  double right_optimal, right_right, right_left, right_total, right_left_mean;
  int right_leftn, right_rightn;

  double left_optimal = 1E32;
  if(left->total > 1)
    left_optimal = left->optimal_sum;

  right_optimal = right_right = right_left = right_total = 1E32;
  right_left_mean = right_leftn = right_rightn = 0;

  /* Right child counts */
  if(right) {
    if(right->total > 1)
      right_optimal = right->optimal_sum;
    right_left = right->left_sum;
    right_leftn = right->leftn;
    right_left_mean = right->left_mean;
  }

  double left_right_right_left_sum = get_aggregate_sum_sq_error(left->right_sum, right_left, left->right_mean, right_left_mean, left->rightn, right_leftn);
  double left_right_agg_mean = get_aggregate_mean(left->right_mean, right_left_mean, left->rightn, right_leftn);

  double left_band = get_confidence_band(left->optimal_mean, left_optimal, left->optimaln);
  double right_band = 1E32;
  double left_right_band = 1E32;
  if(right) {
    right_band = get_confidence_band(right->optimal_mean, right_optimal, right->optimaln);
    left_right_band = get_confidence_band(left_right_agg_mean, left_right_right_left_sum, left->rightn + right_leftn);
  }
  
  int minimum_index = Helper::minimum_of_3(left_band, right_band, left_right_band);
  return minimum_index;
}

/*
 * If is_high = true (Looking for high mean boxes
 *  - Returns the path with highest lower bound for the mean.
 * Else (Looking for low mean boxes
 *  - Returns the path with lowest higher bound for the mean
 */
int numeric_binary_tree::get_optimal_mean_path(numeric_binary_tree *left, numeric_binary_tree *right, bool is_high) {
  double right_optimal, right_right, right_left, right_total, right_left_mean;
  int right_leftn, right_rightn;

  double left_optimal = 1E32;
  if(left->total > 1)
    left_optimal = left->optimal_sum;

  right_optimal = right_right = right_left = right_total = 1E32;
  right_left_mean = right_leftn = right_rightn = 0;

  /* Right child counts */
  if(right) {
    if(right->total > 1)
      right_optimal = right->optimal_sum;
    right_left = right->left_sum;
    right_leftn = right->leftn;
    right_left_mean = right->left_mean;
  }

  double left_right_right_left_sum = get_aggregate_sum_sq_error(left->right_sum, right_left, left->right_mean, right_left_mean, left->rightn, right_leftn);
  double left_right_agg_mean = get_aggregate_mean(left->right_mean, right_left_mean, left->rightn, right_leftn);

  double left_band = get_confidence_band(left->optimal_mean, left_optimal, left->optimaln);
  double right_band = 1E32;
  double left_right_band = 1E32;
  double right_optimal_mean = 0;
  if(right) {
    right_band = get_confidence_band(right->optimal_mean, right_optimal, right->optimaln);
    left_right_band = get_confidence_band(left_right_agg_mean, left_right_right_left_sum, left->rightn + right_leftn);
    right_optimal_mean = right->optimal_mean;
  }
  
  int index = 0;
  if(is_high == true)
    index = Helper::maximum_of_3(left->optimal_mean-left_band, right_optimal_mean-right_band, left_right_agg_mean-left_right_band);
  else
    index = Helper::minimum_of_3(left->optimal_mean+left_band, right_optimal_mean+right_band, left_right_agg_mean+left_right_band);

  return index;
}

/*
 * Update a parent node.
 * Vx (optimal_sum) = min(VLx, VRx, VLr + VRl)
 * Vtotal (total_sum) = VLtotal + VRtotal
 * Modes are -
 * 0 : Low variance
 * 1 : High mean
 * 2 : Low mean
 */
void numeric_binary_tree::update_node() {
  double left_optimal;
  double right_optimal, right_right, right_left, right_total, right_left_mean, right_mean;
  int right_leftn, right_rightn, rightt;

  numeric_binary_tree *left = (numeric_binary_tree *)this->left_child;
  numeric_binary_tree *right = (numeric_binary_tree *)this->right_child;

  left_optimal = 1E32;
  right_optimal = right_right = right_left = right_total = 1E32;
  right_left_mean = right_mean = 0;
  right_leftn = right_rightn = rightt = 0;
  
  if(left->total > 1)
    left_optimal = left->optimal_sum;

  /* Right child counts */
  if(right) {
    if(right->total > 1)
      right_optimal = right->optimal_sum;
    right_total = right->total_sum;
    right_left = right->left_sum;
    right_right = right->right_sum;
    right_leftn = right->leftn;
    right_rightn = right->rightn;
    right_left_mean = right->left_mean;
    rightt = right->total;
    right_mean = right->mean;
  }

  int path = 0;
  switch(this->mode) {
  case 0: // Low variance
    path = get_optimal_path(left, right);
    break;
  case 1: // High mean
    path = get_optimal_mean_path(left, right, true);
    break;
  case 2: // Low mean
    path = get_optimal_mean_path(left, right, false);
    break;
  }

  /* Update optimal start, end indices based on which option gave min sum */

  switch(path) {
  case 0: 
    { // Left
      this->optimal_start = left->optimal_start;
      this->optimal_end = left->optimal_end;
      this->optimal_mean = left->optimal_mean;
      this->optimaln = left->optimaln;
      this->optimal_sum = left_optimal;
      this->left_sum = left->left_sum;
      this->left_mean = left->left_mean;
      this->leftn = left->leftn;
      double right_total_left_right_mean = get_aggregate_mean(right_mean, left->right_mean, rightt, left->rightn);
      double right_total_left_right_sum = get_aggregate_sum_sq_error(right_total, left->right_sum, right_mean, left->right_mean, rightt, left->rightn);
      this->right_sum = right_total_left_right_sum;
      this->right_mean = right_total_left_right_mean;
      this->rightn = rightt + left->rightn;
    }
    break;
  case 1: // Right
    if(right) {
      this->optimal_start = right->optimal_start;
      this->optimal_end = right->optimal_end;
      this->optimal_mean = right->optimal_mean;
      this->optimaln = right->optimaln;
      this->optimal_sum = right_optimal;
      double left_total_right_left_mean = get_aggregate_mean(left->mean, right_left_mean, left->total, right_leftn);
      double left_total_right_left_sum = get_aggregate_sum_sq_error(left->total_sum, right_left, left->mean, right_left_mean, left->total, right_leftn);
      this->left_sum = left_total_right_left_sum;
      this->left_mean = left_total_right_left_mean;
      this->leftn = left->total + right_leftn;
      this->right_sum = right_right;
      this->right_mean = right->right_mean;
      this->rightn = right->rightn;
    }
    break;
  case 2: // Left_right + right_left
    {
      this->optimal_start = left->optimal_start;
      if(right)
	this->optimal_end = right->optimal_end;
      
      double left_right_right_left_sum = get_aggregate_sum_sq_error(left->right_sum, right_left, left->right_mean, right_left_mean, left->rightn, right_leftn);
      double left_right_agg_mean = get_aggregate_mean(left->right_mean, right_left_mean, left->rightn, right_leftn);
      this->optimal_mean = left_right_agg_mean;
      this->optimaln = left->rightn + right_leftn;
      this->optimal_sum = left_right_right_left_sum;
      
      double left_total_right_left_mean = get_aggregate_mean(left->mean, right_left_mean, left->total, right_leftn);
      double left_total_right_left_sum = get_aggregate_sum_sq_error(left->total_sum, right_left, left->mean, right_left_mean, left->total, right_leftn);
      this->left_sum = left_total_right_left_sum;
      this->left_mean = left_total_right_left_mean;
      this->leftn = left->total + right_leftn;
      
      double right_total_left_right_mean = get_aggregate_mean(right_mean, left->right_mean, rightt, left->rightn);
      double right_total_left_right_sum = get_aggregate_sum_sq_error(right_total, left->right_sum, right_mean, left->right_mean, rightt, left->rightn);
      this->right_sum = right_total_left_right_sum;
      this->right_mean = right_total_left_right_mean;
      this->rightn = rightt + left->rightn;
    }
    break;
  }
  
  // Updating totals
  this->total_sum = get_aggregate_sum_sq_error(left->total_sum, right_total, left->mean, right_mean, left->total, rightt);
  this->mean = get_aggregate_mean(left->mean, right_mean, left->total, rightt);
  this->total = left->total + (right ? right->total : 0);

  left->reset_node(true);
  if(right)
    right->reset_node(true);
}

/*
 * Function to insert new value
 * Incrementally updates mean and sum-of-squared-error
 */
void numeric_binary_tree::insert_entry(double score) {
  if(this->total == 0)
    this->total_sum = 0;
  double mean = this->mean + (score - this->mean)/(this->total + 1);
  double sum_sq_error = this->total_sum + (score - mean)*(score - this->mean);
  this->total++;
  
  this->mean = mean;
  this->left_mean = mean;
  this->right_mean = mean;
  this->optimal_mean = mean;
  
  this->total_sum = sum_sq_error;
  this->left_sum = sum_sq_error;
  this->right_sum = sum_sq_error;
  this->optimal_sum = sum_sq_error;
  
  this->leftn++;
  this->rightn++;
  this->optimaln++;
}

/*
 * Print out tree contents for debugging purposes
 */
void numeric_binary_tree::print_tree(Datset *ds, std::vector<int> &iv, int att) {
  printf("%d-%d, (%d-%d), %e, %e, %d, %d, %f,%d,%d\n", this->first, this->last, this->optimal_start, this->optimal_end, this->optimal_sum,
         this->total_sum, this->total, this->optimaln, this->optimal_mean, this->leftn, this->rightn);
  if(left_child)
    left_child->print_tree(ds, iv, att);
  if(right_child)
    right_child->print_tree(ds, iv, att);
}
