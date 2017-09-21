/*
   File:        discrete_binary_tree.hpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 19 16:23:48 EST 2016
   Description: Binary tree for finding maximum sum
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef DISCRETE_BINARY_TREE_H_
#define DISCRETE_BINARY_TREE_H_

#include "binary_tree.hpp"

/*
 * Binary tree data structure for representing all the values of a dimension at the leaves.
 * The tree has aggregated counts representing which contiguous block of indices (start - end) result in highest summing box (positives - negatives).
 */
class discrete_binary_tree : public btree_node {
private:
  int total_pos, total_neg;        /* Total pos and neg labels at that node */
  int opt_pos, opt_neg;            /* Total optimal pos and neg labels at that node */
  int left_pos, left_neg;          /* Total pos and neg labels at that node for left optimal half */
  int right_pos, right_neg;        /* Total pos and neg labels at that node for right optimal half */
  std::vector<int> label_dyv;      /* NULL for non-leaf nodes. Contains class distribution of points in this leaf */
 public:
  std::vector<int> & get_label_dyv();
 
  std::string get_name();

  int get_total() { return total_pos + total_neg; }

  /*
   * Reset tree counts 
   */
  void reset_node(bool exclude_leaves);

  int get_opt_pos() { return opt_pos; }
  int get_opt_neg() { return opt_neg; }

  /*
   * Update a parent node.
   * Vx (optimal_sum) = max(VLx, VRx, VLr + VRl)
   * Vl (left_sum) = max(VLl, VLtotal + VRl)
   * Vr (right_sum) = max(VRr, VRtotal + VLr)
   * Vtotal (total_sum) = VLtotal + VRtotal
   */
  void update_node();

  /*
   * Constructor
   */
  discrete_binary_tree(int first, int last, double end_value, int arity);

  /*
   * Constructor for creating parent node from child nodes
   * right_child may be NULL
   */
  discrete_binary_tree(btree_node *left_child, btree_node *right_child);

  /*
 * Destructor
 */
  ~discrete_binary_tree();

  /*
   * Traverse through all leaf nodes (without traversing entire tree)
   * Change pos = counts for 'class' type.
   * Change neg = counts for other types.
   */
  static void set_leaves_pos(std::vector<btree_node *> & leaves, int class_label);

  /*
   * Function to insert new value at the correct leaf node
   */
  void insert_entry(double score);

  /*
   * Print out tree contents
   */
  void print_tree(Datset *ds, std::vector<int> &iv, int att);
};

#endif
