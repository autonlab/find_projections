/*
   File:        numeric_binary_tree.hpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Binary tree for finding low variance(low stderror) regions
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef NUMERIC_BINARY_TREE_H_
#define NUMERIC_BINARY_TREE_H_

#include "binary_tree.hpp"

/*
 * Binary tree for finding low variance(low stderror) regions
 */
class numeric_binary_tree : public btree_node {
private:
  int mode;
  int total;
  double mean, left_mean, right_mean, optimal_mean;
  int leftn, rightn, optimaln;
  static int get_optimal_path(numeric_binary_tree *left, numeric_binary_tree *right);
  static int get_optimal_mean_path(numeric_binary_tree *left, numeric_binary_tree *right, bool is_high);
 public:
  int get_optimal_total() { return optimaln; }
  int get_total() { return total; }
  double get_mean() { return optimal_mean; }

  /*
   * 0 : Low variance
   * 1 : High mean
   * 2 : Low mean
   */
  void set_mode(int mode) {
    this->mode = mode;
    if(left_child) 
      ((numeric_binary_tree *)left_child)->set_mode(mode);
    if(right_child) 
      ((numeric_binary_tree *)right_child)->set_mode(mode);
  }

  std::string get_name();
  
  /*
   * Reset tree counts 
   */
  void reset_tree(bool exclude_leaves);
  
  void reset_node(bool exclude_leaves);

  /*
   * Destructor
   */
  ~numeric_binary_tree();

  /*
   * Constructor
   */
  numeric_binary_tree(int first, int last, double end_value);
  
  /*
   * Constructor for creating parent node from child nodes
   * right_child may be NULL
   */
  numeric_binary_tree(btree_node *left_child, btree_node *right_child);

  /*
   * Update a parent node.
   * Vx (optimal_sum) = min(VLx, VRx, VLr + VRl)
   * Vtotal (total_sum) = VLtotal + VRtotal
   */
  void update_node();

  /*
   * Function to insert new value
   * Incrementally updates mean and sum-of-squared-error
   */
  void insert_entry(double score);

  /*
   * Print out tree contents for debugging purposes
   */
  void print_tree(Datset *ds, std::vector<int> &iv, int att);
};

#endif
