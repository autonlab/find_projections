/*
   File:        binary_tree.hpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 19 16:23:48 EST 2016
   Description: Binary tree for finding maximum sum/minimum variance projection boxes
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef TREE_H_
#define TREE_H_

#include <vector>
#include <string>
#include "datset.hpp"

/*
 * Binary tree data structure for representing all the values of a dimension at the leaves.
 * The tree has aggregated counts representing which contiguous block of indices (start - end) result in highest summing box (positives - negatives).
 */
class btree_node {
 protected:
  int optimal_start, optimal_end;  /* Optimal start - end range indices */
  double optimal_sum;              /* Optimal sum within optimal range = (Pos - neg) */
  double total_sum;                /* Total sum of all data at the node */
  int first, last;                 /* Range of indices valid for a node */
  double left_sum;                 /* Sum from first : optimal_end for a node */
  double right_sum;                /* Sum from optimal_start : last for a node */
  btree_node *left_child;          /* Pointer to left child (May be NULL) */ 
  btree_node *right_child;         /* Pointer to right child (May be NULL) */

  /*
   * Returns FALSE for
   * 1) Leaf node
   * 2) Total counts at node = 0
   * Else returns TRUE
   */
  bool valid_to_update() {
    if(left_child == NULL)
      return false;
    
    if(this->left_child->get_total() == 0) {
      if(!this->right_child)
	return false;
      else if(this->right_child->get_total() == 0)
	return false;
    }
    return true;
  }

  bool is_leaf() { return left_child == NULL; }
 public:
  double right_cutoff;             /* Cutoff value corresponding to highest element in this bin */

  /* Returns the optimal start index for the entire tree */
  int get_node_optimal_start();
  
  /* Returns the optimal end index for the entire tree */
  int get_node_optimal_end();
  
  /* Returns the optimal sum for the entire tree */
  double get_node_optimal_sum();

  virtual std::string get_name() = 0;

  /*
   * Creates the entire tree upfront for all values for the attribute att
   * iv - Subset of rows for which tree is being constructed (in sorted order by the values of attribute)
   * bin_size - No. of data points at each leaf (Maybe more if we have very close values around)
   */
  static btree_node *construct_empty_tree(Datset &ds, int att, bool is_classifier, std::vector<int> &train_rows, std::vector<int> &iv,
                                          int bin_size, std::vector<btree_node *> *leaves);
  
  /*
   * Reset tree counts 
   */
  virtual void reset_node(bool exclude_leaves) = 0;

  virtual int get_total() = 0;

  virtual void update_node() = 0;

  virtual void print_tree(Datset *ds, std::vector<int> &iv, int att) = 0;

  /*
   * Updates entire tree calling update_node() on non-leaf nodes to update their optimal counts, range
   * After this function completes, the root node will have the optimal counts, range computed for the data contained in this tree
   * Post-order traversal
   */
  void update_tree() {
    if(left_child)
      left_child->update_tree();
    if(right_child)
      right_child->update_tree();
    
    if(valid_to_update() == false)
      return;

    update_node();
  }

  virtual ~btree_node();

  /*
   * Constructor
   */
  btree_node(int first, int last, double end_value);

  /*
   * Constructor for creating parent node from child nodes
   * right_child may be NULL
   */
  btree_node(btree_node *left_child, btree_node *right_child);

  virtual void insert_entry(double score) = 0;

  static void insert_in_leaf_node(std::vector<btree_node *> & leaves, double value, double score);
};

#endif
