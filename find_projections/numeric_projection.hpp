/*
   File:        numeric_projection.hpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 19 16:23:48 EST 2016
   Description: Represents 2-d subset of data (For low variance)
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef NUMERIC_PROJECT_H_
#define NUMERIC_PROJECT_H_

#include "projection.hpp"

/*
 * Represents 2-d subset of data, bounded like a rectangular box.
 */
class numeric_projection : public projection {
 private:
  int total;
  double mean, sum_sq_error;
 public:
  const int get_total() { return total; }
  void set_total(int total) { this->total = total; }

  const double get_mean() { return mean; }
  void set_mean(double mean) { this->mean = mean; }

  const double get_sum_sq_error() { return sum_sq_error; }
  void set_sum_sq_error(double sum_sq_error) { this->sum_sq_error = sum_sq_error; }

  void pprojection() {
    printf("Projection for dims (%d, %d) at %f : %f and %f : %f, %d, %f, %e\n", this->att1, this->att2, this->att1_start,
	   this->att1_end, this->att2_start, this->att2_end, this->total, this->mean, this->sum_sq_error);
  }

  void pprojection_for_file(FILE *fp) {
    fprintf(fp, "%d,%d,%f,%f,%f,%f,%d,%f,%e\n", this->att1, this->att2, this->att1_start,
	    this->att1_end, this->att2_start, this->att2_end, this->total, this->mean, this->sum_sq_error);
  }

  void print_header_row(FILE *fp) {
    fprintf(fp, "Dim1,Dim2,xmin,xmax,ymin,ymax,Total,Mean,Sum-Sq-Error\n");
  }

  ~numeric_projection() {}

  /*
   * Returns true if projection 'qr' is better than projection 'pr'
   * Better is defined by lower sum_sq_error or higher support(counts in projection) if sum_sq_error are equal.
   */
  bool is_projection_better(projection *qr) {
    numeric_projection *npr = (numeric_projection *)qr;
    double score1 = this->sum_sq_error;
    double score2 = npr->sum_sq_error;
    
    if(score1 == score2)
      return(npr->total > this->total);
    else
      return (score2 < score1);
  }

  bool is_projection_good_on_validation_set(Datset &ds, std::vector<int> &val_rows, int mode, double mean, double purity, int *support) {
    double result = apply_projection_on_ds(ds, val_rows, support); 
  
    switch(mode) {
    case 0:
      return true;
    case 1:
      return (result >= mean);
    case 2:
      return (result <= mean);
    }

    return false;
  }

  void copy_projection(projection *pr) {
    numeric_projection *npr = (numeric_projection *)pr;
    projection::copy_projection(pr);
    npr->total = this->total;
    npr->mean = this->mean;
    npr->sum_sq_error = this->sum_sq_error;
  }

  double apply_projection_on_ds(Datset &ds, std::vector<int> &valid_rows, int *support) {
    int i;
    int rows = valid_rows.size() ? valid_rows.size() : ds.get_rows();
    int count = 0;
    double sum = 0.0;

    for(i=0; i<rows; i++) {
      int r = valid_rows.size()>0 ? valid_rows[i] : i;
      double label = ds.ds_output_ref(r);
      
      if(this->point_lies_in_projection(ds, r) == true) {
	    count++;
        sum += label;
      }
    }
    
    *support = count;
    return (double)sum/(double)count;
  }

  numeric_projection(int att1, int att2, double att1_start, double att1_end, double att2_start, double att2_end) :
    projection(att1, att2, att1_start, att1_end, att2_start, att2_end) {
  }

  numeric_projection() {
  }

  /*
   * Returns true/false for whether query point 'row' if point lies in projection.
   * If true, 'mean' will contain the mean of that projection box.
   */
  bool score_predictions(Datset &ds, int row, double *mean) {
    bool flag = point_lies_in_projection(ds, row);
    
    if(!flag) 
      return false;
    
    *mean = this->mean;
    return true;
  }

  /*
   * Compute R2 of data points in this projection box
   */
  double compute_R2(Datset &ds, std::vector<int> &train_rows) {
    double sum = 0, R2;
    double predicted_mean = mean;
    unsigned int i, testrows = indices->size();
    double variance = 0.0;

    double truemean = 0.0;
    for(i=0; i<train_rows.size(); i++) {
      int row = train_rows[i];
      double value = ds.ds_output_ref(row);
      truemean += value;
    }
    truemean /= train_rows.size();

    for(i=0; i<testrows; i++) {
      int row = train_rows[indices->at(i)];
      double value = ds.ds_output_ref(row);
      sum += pow(predicted_mean - value, 2);
      variance += pow(value - truemean, 2);
    }
   
    R2 = 1.0 - (sum / variance);
    return R2;
  }

  /*
   * Returns the projection metric (class label for discrete_projection / mean for numeric_projection)
   */
  double get_projection_metric() {
    return mean;
  }
};

#endif
