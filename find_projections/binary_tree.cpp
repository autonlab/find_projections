/*
   File:        binary_tree.cpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 19 16:23:48 EST 2016
   Description: 
   Copyright (c) 2016 Carnegie Mellon University
*/

#include "binary_tree.hpp"
#include "discrete_binary_tree.hpp"
#include "numeric_binary_tree.hpp"
#include "helper.hpp"

using namespace std;

/*
 * Constructor
 */
btree_node::btree_node(int first, int last, double end_value):
  first(first), last(last), right_cutoff(end_value) {

  this->optimal_start = first;
  this->optimal_end = last;

  this->left_child = NULL;
  this->right_child = NULL;
}

/*
 * Constructor for creating parent node from child nodes
 * right_child may be NULL
 */
btree_node::btree_node(btree_node *left_child, btree_node *right_child) :
  left_child(left_child), right_child(right_child) {
  this->first = left_child->first;
  this->last = right_child ? right_child->last : left_child->last;

  this->optimal_start = this->first;
  this->optimal_end = this->last;

  this->right_cutoff = right_child ? right_child->right_cutoff : left_child->right_cutoff;
}

/*
 * Creates the entire tree upfront for all values for the attribute att
 * iv - Subset of rows for which tree is being constructed
 * bin_size - No. of data points at each leaf (Maybe more if we have very close values around)
 */
btree_node *btree_node::construct_empty_tree(Datset &ds, int att, bool is_classifier, std::vector<int> &train_rows, std::vector<int> &iv,
                                             int bin_size, std::vector<btree_node *> *leaves) {
  unsigned int i, start = 0, end = iv.size()-1;
  unsigned int size = end-start+1;
  std::vector<btree_node *> garr;
  garr.reserve(size/bin_size);

  i = start;
  unsigned int binend = start;
  int arity = ds.get_num_classes();

  /* Construct all the leaf nodes first */
  while(i <= end) {
    binend = i+bin_size-1;
    if(binend > end)
      binend = end;

    double end_value = ds.ds_real_ref(train_rows[iv[binend]], att);

    /* Try to advance index for duplicates/very very close values */
    for(unsigned int k=binend+1; k<=end; k++) {
      double value = ds.ds_real_ref(train_rows[iv[k]], att);
      if(value - end_value < MIN_DOUBLE) {
	end_value = value;
	binend++;
      }
      else
	break;
    }

    btree_node *node = NULL;
    if(is_classifier)
      node = new discrete_binary_tree(i, binend, end_value, arity);
    else
      node = new numeric_binary_tree(i, binend, end_value);
    garr.push_back(node);
    i = binend+1;
  } /* All leaves created */

    /* Contents should NEVER be deleted from this array. This array is just for browsing-the-leaves purposes */
  leaves->reserve(garr.size());
  for ( i = 0 ; i < garr.size() ; i++ ) {
    leaves->push_back(garr[i]);
  }

  /* Build parent nodes from leaf nodes till we reach root node 
   * Right child may be NULL for some parent nodes
   */
  while(garr.size() > 1) {
    std::vector<btree_node *> garr2;
    garr2.reserve(garr.size()/2+1);

    for(i=0; i<garr.size(); i+=2) {
      btree_node *left_child = garr[i];
      btree_node *right_child = (i+1 < garr.size()) ? garr[i+1] : NULL;
      btree_node *node = NULL;
      if(is_classifier)
	node = new discrete_binary_tree(left_child, right_child);
      else
	node = new numeric_binary_tree(left_child, right_child);
      garr2.push_back(node);
    }

    garr = garr2;
  }

  /* Return root node */
  btree_node *node = garr[0];

  return node;
}

/* Returns the optimal sum for the entire tree */
double btree_node::get_node_optimal_sum() {
  return optimal_sum;
}

/*
 * Destructor
 */
btree_node::~btree_node() {
  if(left_child)
    delete left_child;
  if(right_child)
    delete right_child;
}

/* Returns the optimal start index for the entire tree */
int btree_node::get_node_optimal_start() {
  return this->optimal_start;
}

/* Returns the optimal end index for the entire tree */
int btree_node::get_node_optimal_end() {
  return this->optimal_end;
}

/*
 * Binary search to find appropriate leaf (data bin) to insert value
 */
static unsigned int find_leaf(std::vector<btree_node *> & leaves, double value) {
  unsigned int lb = 0, ub = leaves.size()-1;

  while(lb < ub) {
    int M = (lb + ub)/2;
    double cutoff = leaves[M]->right_cutoff;
    
    if(value <= cutoff) { // Match
      if(M == 0)
	return M;
      if(leaves[M-1]->right_cutoff < value)
	return M;
      else 
	ub = M-1;
      continue;
    }
    else
      lb = M+1;
  }

  return lb; 
}

void btree_node::insert_in_leaf_node(std::vector<btree_node *> & leaves, double value, double score) {
  unsigned int index = find_leaf(leaves, value);
  btree_node *btn = leaves[index];
  btn->insert_entry(score);
}
