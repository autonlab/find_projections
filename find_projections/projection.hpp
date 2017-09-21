/*
   File:        projection.hpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Represents 2-d subset of data
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef PROJECT_H_
#define PROJECT_H_

#include "binary_tree.hpp"
#include "helper.hpp"

/*
 * Represents 2-d subset of data, bounded like a rectangular box.
 */
class projection {
 protected:
  int att1, att2;                  /* Attributes in the projection */
  double att1_start, att1_end;     /* Att 1's start and end values of the box */
  double att2_start, att2_end;     /* Att 2's start and end values of the box */
  std::vector<int> *indices;                   /* Indices of the entire dataset inside this projection */

 public:
  /*
   * Constructor
   */
  projection(int att1, int att2, double att1_start, double att1_end, double att2_start, double att2_end);

  projection();

  const int get_att1() { return att1; }
  void set_att1(int att1) { this->att1 = att1; }

  const int get_att2() { return att2; }
  void set_att2(int att2) { this->att2 = att2; }

  const double get_att1_start() { return att1_start; }
  void set_att1_start(double att1_start) { this->att1_start = att1_start; }

  const double get_att1_end() { return att1_end; }
  void set_att1_end(double att1_end) { this->att1_end = att1_end; }

  const double get_att2_start() { return att2_start; }
  void set_att2_start(double att2_start) { this->att2_start = att2_start; }

  const double get_att2_end() { return att2_end; }
  void set_att2_end(double att2_end) { this->att2_end = att2_end; }

  std::vector<int> *get_indices() {
    return indices;
  }

  virtual void pprojection() = 0;

  virtual void pprojection_for_file(FILE *fp) = 0;

  virtual ~projection();

  virtual void print_header_row(FILE *fp) = 0;

  virtual double apply_projection_on_ds(Datset &ds, std::vector<int> &valid_rows, int *support) = 0;

  virtual bool is_projection_good_on_validation_set(Datset &ds, std::vector<int> &val_rows, int mode, double mean, double purity, int *support) = 0;

  /*
   * Returns true if projections are for same pair of attributes and have overlapping boxes
   */
  virtual bool do_projections_overlap(projection *pr2);

  /*
   * Returns true if projection 'qr' is better than projection 'pr'
   */
  virtual bool is_projection_better(projection *qr) = 0;

  virtual void copy_projection(projection *pr);

  /*
   * Construct dataset indices falling in this projection
   */
  void mk_projection_indices(Datset &ds, std::vector<int> &train_rows, indices_array &ia);

  /*
   * Returns true if query point lies inside projection
   */
  bool point_lies_in_projection(Datset &ds, int row);

  /*
   * Returns the projection metric (class label for discrete_projection / mean for numeric_projection)
   */
  virtual double get_projection_metric() = 0;

  /*
   * Creates projection from -
   * att1's row subset (start-end) of ivatt1
   * att2's row subset selected in binary tree 'node' of ivatt2
   * node - Binary tree containing optimal range of att2
   */
  static projection *mk_projection_from_tree(btree_node *node, Datset &ds, std::vector<int> &train_rows, 
					     std::vector<int> &ivatt1, int start, int end,
					     std::vector<int> &ivatt2, int att1, int att2);

  virtual const int get_total() = 0;

  static void print_decision_list(std::vector<projection *> garr, Datset &ds, std::vector<double> &proportions) {
    char buf1[10000], buf2[10000];
    for(unsigned int i=0; i<garr.size(); i++) {
      projection *pr = garr[i];
      sprintf(buf1, "x%d", pr->get_att1());
      sprintf(buf2, "x%d", pr->get_att2());
      double accuracy = proportions[i] * 100;
      const char *keyword = (i == 0)?"IF":"ELSE IF";
      printf("%s %f <= %s <= %f AND %f <= %s <= %f, Coverage = %0.2f percent\n", keyword,
	     pr->get_att1_start(), buf1, pr->get_att1_end(), pr->get_att2_start(), buf2, pr->get_att2_end(), accuracy);
    }
    printf("\n");
  }
};

#endif
