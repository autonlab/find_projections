/*
   File:        feature_tree.hpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Class for storing binary tree data structure for each feature
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef FTREE_H_
#define FTREE_H_

#include "binary_tree.hpp"
#include <vector>

class indices {
private:
  int m, n;
public:
  indices(int m, int n) : m(m), n(n) {
  }

  int getm() { return m;}
  int getn() { return n;}
};

/*
 * Class for storing binary tree data structure for each feature
 */
class feature_tree {
private:
  int atts;
  std::vector<btree_node *> table;
  std::vector<std::vector<btree_node *> *> leaves;
  std::vector<std::vector<indices> *> indices_vec;
public:
  feature_tree(int atts) {
    this->atts = atts;
    table.resize(atts);
    leaves.resize(atts);
    indices_vec.resize(atts);
  }
  
  ~feature_tree() { 
    for(unsigned int j=0; j<table.size(); j++) {
      btree_node *btn = table[j];
      delete btn;
      delete leaves[j];
      delete indices_vec[j];
    }
  }
    
  btree_node *getTree(int i) {
    return table[i];
  }
  
  void setTree(int i, btree_node *btn) {
    btree_node **ptr = table.data();
    ptr[i] = btn;
  }

  std::vector<btree_node *> *getLeaves(int i) {
    return leaves[i];
  }
   
  void setLeaves(int i, std::vector<btree_node *> *leaves) {
    this->leaves[i] = leaves;
  }

  std::vector<indices> *getIndices(int i) {
    return indices_vec[i];
  }
   
  void setIndices(int i, std::vector<indices> *indices_vec) {
    this->indices_vec[i] = indices_vec;
  }
};

#endif
