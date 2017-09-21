/*
   File:        feature_map.hpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Class for storing list of projections for each feature-pair
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef FMAP_H_
#define FMAP_H_

#include "projection.hpp"
#include <vector>

/*
 * Container for all the projection boxes found in a pair of attributes.
 * Does not own memory of its projection members
 */
class prlist {
private:
  std::vector<projection *>prvec;
public:
  int size() { return prvec.size(); }
  projection *get(int i) { return prvec[i]; }
  std::vector<projection *> getList() { return prvec; }
  prlist(std::vector<projection *> & prvec) { this->prvec = prvec; }
  prlist() {}
};

/*
 * Container for all the projection boxes found in the data.
 */
class feature_map {
private:
  int atts;
  std::vector<prlist> table;
public:
  feature_map(int atts) {
    this->atts = atts;
    int total = atts * atts;
    table.resize(total);
  }
  
  feature_map() {
  }

  ~feature_map() { 
    for(unsigned int j=0; j<table.size(); j++) {
      prlist vec = table[j];
      for(int i = vec.size()-1; i>=0; i--) {
	projection *pr = vec.get(i);
	delete pr;
      }
    }
  }
    
  prlist& getListOfProjections(int i, int j) {
    int index = i*atts + j;
    return table[index];
  }
  
  void setProjections(int i, int j, prlist & pr_array) {
    prlist *ptr = table.data();
    int index = i *atts + j;
    ptr[index] = pr_array;
  }

  void print_projections() {
    for(unsigned int j=0; j<table.size(); j++) {
      prlist vec = table[j];
      for(int i = 0; i<vec.size(); i++) {
	projection *pr = vec.get(i);
	pr->pprojection();
      }
    }
  }

  int get_num_projections() {
    int sum = 0;
    for(unsigned int j=0; j<table.size(); j++) {
      prlist vec = table[j];
      sum += vec.size();
    }
    return sum;
  }

  projection *get_projection(int i) {
    if(i < 0)
      return NULL;

    int sum = 0;
    projection *pr = NULL;
    for(unsigned int j=0; j<table.size(); j++) {
      prlist vec = table[j];
      if(vec.size() == 0)
        continue;
      if(i >= sum && i < sum + vec.size()) {
        pr = vec.get(i-sum);
        break;
      }
      sum += vec.size();
    }

    return pr;
  }

  void print_projections_to_file(FILE *fp) {
    bool first_printed = false;
    for(unsigned int j=0; j<table.size(); j++) {
      prlist vec = table[j];
      for(int i = 0; i<vec.size(); i++) {
	projection *pr = vec.get(i);
	if(!first_printed) {
	  pr->print_header_row(fp);
	  first_printed = true;
	}
	pr->pprojection_for_file(fp);
      }
    }
  }
};

/*
 * Container for all the projection boxes found in the data.
 * Owns memory of its projection members
 */
class projection_array {
private:
  std::vector<projection *> vec;
public:
  projection_array() {
  }
  
  projection_array(std::vector<projection *> & prvec) {
    this->vec = prvec;
  } 

  ~projection_array() {
    for(int i = vec.size()-1; i>=0; i--) {
      projection *pr = vec[i];
      delete pr;
    }
  }

  void print_projections_to_file(FILE *fp) {
    bool first_printed = false;
    for(unsigned int i = 0; i<vec.size(); i++) {
      projection *pr = vec[i];
      if(!first_printed) {
        pr->print_header_row(fp);
        first_printed = true;
      }
      pr->pprojection_for_file(fp);
    }
  }

  int get_num_projections() {
    return vec.size();
  }

  projection *get_projection(unsigned int i) {
    if(i < 0 || i >= vec.size())
      return NULL;
    return vec[i];
  }
};

#endif
