/*
   File:        discrete_binary_tree.cpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Binary tree for finding maximum sum
   Copyright (c) 2016 Carnegie Mellon University
*/

#include "discrete_binary_tree.hpp"
#include "helper.hpp"

/*
 * Constructor
 */
discrete_binary_tree::discrete_binary_tree(int first, int last, double end_value, int arity) : btree_node(first, last, end_value) {
  this->total_pos = this->total_neg = 0;
  this->opt_pos = this->opt_neg = this->left_pos = this->left_neg = this->right_pos = this->right_neg = 0;
  this->label_dyv.resize(arity);
  this->total_sum = this->left_sum = this->right_sum = this->optimal_sum = 0;
}

std::string discrete_binary_tree::get_name() {
  return std::string("discrete_binary_tree");
}

/*
 * Constructor for creating parent node from child nodes
 * right_child may be NULL
 */
discrete_binary_tree::discrete_binary_tree(btree_node *left_child, btree_node *right_child) : btree_node(left_child, right_child) {
  this->total_pos = this->total_neg = 0;
  this->opt_pos = this->opt_neg = this->left_pos = this->left_neg = this->right_pos = this->right_neg = 0;
  this->total_sum = this->left_sum = this->right_sum = this->optimal_sum = 0;
}

std::vector<int> & discrete_binary_tree::get_label_dyv() {
  return label_dyv;
}

/*
 * Reset tree counts 
 */
void discrete_binary_tree::reset_node(bool exclude_leaves) {
  this->total_sum = 0;
  this->optimal_sum = 0;
  this->left_sum = this->right_sum = 0;
  this->optimal_start = this->first;
  this->optimal_end = this->last;

  this->total_pos = this->total_neg = 0;
  this->opt_pos = this->opt_neg = this->left_pos = this->left_neg = this->right_pos = this->right_neg = 0;

  if(exclude_leaves == false) {
    for(unsigned int j=0; j<label_dyv.size(); j++)
      label_dyv[j] = 0;
  }
}

/*
 * Destructor
 */
discrete_binary_tree::~discrete_binary_tree() {
}

/*
 * Update a parent node.
 * Vx (optimal_sum) = max(VLx, VRx, VLr + VRl)
 * Vl (left_sum) = max(VLl, VLtotal + VRl)
 * Vr (right_sum) = max(VRr, VRtotal + VLr)
 * Vtotal (total_sum) = VLtotal + VRtotal
 */
void discrete_binary_tree::update_node() {
  double right_optimal, right_right, right_left, right_total;

  right_optimal = right_right = right_left = right_total = 0;

  discrete_binary_tree *left = (discrete_binary_tree *)this->left_child;
  discrete_binary_tree *right = (discrete_binary_tree *)this->right_child;

  /* Right child counts */
  if(right) {
    right_optimal = right->optimal_sum;
    right_total = right->total_sum;
    right_left = right->left_sum;
    right_right = right->right_sum;
  }
  
  int path = Helper::maximum_of_3(left->optimal_sum, right_optimal, left->right_sum + right_left);

  /* Update optimal start, end indices based on which option gave max sum */
  switch(path) {
  case 0: // Left optimal
    this->optimal_sum = left->optimal_sum;
    this->optimal_start = left->optimal_start;
    this->optimal_end = left->optimal_end;
    this->opt_pos = left->opt_pos;
    this->opt_neg = left->opt_neg;
    this->left_pos = left->left_pos;
    this->left_neg = left->left_neg;
    this->right_pos = left->right_pos;
    this->right_neg = left->right_neg;
    if(right) {
      this->right_pos += right->total_pos;
      this->right_neg += right->total_neg;
    }
    break;
  case 1: // Right optimal
    this->optimal_sum = right_optimal;
    this->left_pos = left->total_pos;
    this->left_neg = left->total_neg;
    if(right) {
      this->optimal_start = right->optimal_start;
      this->optimal_end = right->optimal_end;
      this->opt_pos = right->opt_pos;
      this->opt_neg = right->opt_neg;
      this->left_pos += right->left_pos;
      this->left_neg += right->left_neg;
      this->right_pos = right->right_pos;
      this->right_neg = right->right_neg;
    }
    break;
  case 2: // Left_right + right_left
    this->optimal_sum = left->right_sum + right_left;
    this->optimal_start = left->optimal_start;
    this->opt_pos = left->right_pos;
    this->opt_neg = left->right_neg;
    this->left_pos = left->total_pos;
    this->left_neg = left->total_neg;
    this->right_pos = left->right_pos;
    this->right_neg = left->right_neg;
    if(right) {
      this->optimal_end = right->optimal_end;
      this->opt_pos += right->left_pos;
      this->opt_neg += right->left_neg;
      this->left_pos += right->left_pos;
      this->left_neg += right->left_neg;
      this->right_pos += right->total_pos;
      this->right_neg += right->total_neg;
    }
    break;
  }

  // Updating left_sum
  if(left->left_sum >= left->total_sum + right_left) {
    this->left_sum = left->left_sum;
  }
  else {
    this->left_sum = left->total_sum + right_left;
  }

  // Updating right_sum
  if(right_right >= right_total + left->right_sum) {
    this->right_sum = right_right;
  }
  else {
    this->right_sum = right_total + left->right_sum;
  }
  
  // Updating totals
  this->total_sum = left->total_sum + right_total;
  this->total_pos = left->total_pos;
  this->total_neg = left->total_neg;
  if(right) {
    this->total_pos += right->total_pos;
    this->total_neg += right->total_neg;
  }

  left->reset_node(true);
  if(right)
    right->reset_node(true);
}

/*
 * Print out tree contents
 */
void discrete_binary_tree::print_tree(Datset *ds, std::vector<int> &iv, int att) {
  printf("%d-%d, (%d-%d), %d %d, %d %d(%d %d)\n", this->first, this->last, this->optimal_start, this->optimal_end, (int)this->optimal_sum,
         (int)this->total_sum, this->total_pos, this->total_neg, this->opt_pos, this->opt_neg);
  if(left_child)
    ((discrete_binary_tree *)left_child)->print_tree(ds, iv, att);
  if(right_child)
    ((discrete_binary_tree *)right_child)->print_tree(ds, iv, att);
}

static int get_sum(const std::vector<int> & vec) {
  int sum = 0;
  for(unsigned int i=0; i<vec.size(); i++)
    sum += vec[i];
  return sum;
}

/*
 * Traverse through all leaf nodes (without traversing entire tree)
 * Change pos = counts for 'class' type.
 * Change neg = counts for other types.
 */
void discrete_binary_tree::set_leaves_pos(std::vector<btree_node *> & leaves, int class_label) {
  int i=0, gsize = leaves.size();
  for(i=0; i<gsize; i++) {
    discrete_binary_tree *node = (discrete_binary_tree *)leaves[i];
    const std::vector<int> & vec = node->get_label_dyv();
    int pos = vec[class_label];
    int neg = get_sum(vec) - pos;
    int net = pos - neg;
    node->total_sum = net;
    node->optimal_sum = net;
    node->left_sum = net;
    node->right_sum = net;
    node->total_pos = node->right_pos = node->left_pos = node->opt_pos = pos;
    node->total_neg = node->right_neg = node->left_neg = node->opt_neg = neg;
  }
}

/*
 * Function to insert new value at the correct leaf node
 */
void discrete_binary_tree::insert_entry(double score) {
  int label = (int)score;
  (label_dyv[label])++;
}
