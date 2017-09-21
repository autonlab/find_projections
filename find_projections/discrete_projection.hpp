/*
   File:        discrete_projection.hpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: Represents 2-d subset of data (For maximum sum)
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef DISCRETE_PROJECT_H_
#define DISCRETE_PROJECT_H_

#include "projection.hpp"

/*
 * Represents 2-d subset of data, bounded like a rectangular box.
 */
class discrete_projection : public projection {
 private:
  int pos, neg;                    /* No. of pos and neg labels inside the projection */
  int class_label;                 /* Class representing positive in this box */
 public:
  const int get_pos() { return pos; }
  void set_pos(int pos) { this->pos = pos; }

  const int get_neg() { return neg; }
  void set_neg(int neg) { this->neg = neg; }

  const int get_class() { return class_label; }
  void set_class(int class_label) { this->class_label = class_label; }

  const int get_total() { return pos+neg; }

  void pprojection() {
    printf("Projection(class %d) for dims (%d, %d) at %f : %f and %f : %f (%d pos, %d neg, purity = %f)\n", this->class_label, this->att1, this->att2, this->att1_start,
	   this->att1_end, this->att2_start, this->att2_end, this->pos, this->neg, (double)this->pos/(double)(this->pos+this->neg));
  }

  void pprojection_for_file(FILE *fp) {
    fprintf(fp, "%d,%d,%d,%d,%f,%f,%f,%f,%d,%d,%f\n", this->class_label, this->att1, this->att2, this->pos - this->neg, this->att1_start,
	    this->att1_end, this->att2_start, this->att2_end, this->pos, this->neg, (double)this->pos/(double)(this->pos+this->neg));
  }

  void print_header_row(FILE *fp) {
    fprintf(fp, "Class,Dim1,Dim2,Score,xmin,xmax,ymin,ymax,Pos,Neg,Purity\n");
  }

  ~discrete_projection() {}

  /*
   * Returns true if projections are for same class, same pair of attributes and have overlapping boxes
   */
  bool do_projections_overlap(projection *pr2) {
    if(!pr2)
      return false;
    
    discrete_projection *dpr = (discrete_projection *)pr2;
    if(this->class_label != dpr->class_label)
      return false;
    
    bool overlap = projection::do_projections_overlap(pr2);
    return overlap;
  }

  /*
   * Returns true if projection 'qr' is better than projection 'pr'
   * Better is defined by higher purity (=pos/total) or higher support(counts in projection) if purities are equal.
   */
  bool is_projection_better(projection *qr) {
    discrete_projection *dpr = (discrete_projection *)qr;
    double purity1 = (double)this->pos/(double)(this->pos + this->neg);
    double purity2 = (double)dpr->pos/(double)(dpr->pos + dpr->neg);
    
    if(purity1 == purity2) 
      return((dpr->pos + dpr->neg) > (this->pos + this->neg));
    else
      return (purity2 > purity1);
  }

  bool is_projection_good_on_validation_set(Datset &ds, std::vector<int> &val_rows, int mode, double mean, double purity, int *support) {
    double result = apply_projection_on_ds(ds, val_rows, support); 
    
    return (result >= purity);
  }

  void copy_projection(projection *pr) {
    discrete_projection *dpr = (discrete_projection *)pr;
    projection::copy_projection(pr);
    dpr->class_label = this->class_label;
    dpr->pos = this->pos;
    dpr->neg = this->neg;
  }

  discrete_projection(int att1, int att2, double att1_start, double att1_end, double att2_start, double att2_end) :
    projection(att1, att2, att1_start, att1_end, att2_start, att2_end) {
  }

  discrete_projection() {}

  double apply_projection_on_ds(Datset &ds, std::vector<int> &valid_rows, int *support) {
    int i;
    int rows = valid_rows.size()>0 ? valid_rows.size() : ds.get_rows();
    int count = 0, class_count = 0;

    for(i=0; i<rows; i++) {
      int r = valid_rows.size() ? valid_rows[i] : i;
      int label = (int)ds.ds_output_ref(r);
      
      if(this->point_lies_in_projection(ds, r) == true) {
	    count++;
	    if(label == this->class_label)
	      class_count++;
      }
    }
    
    *support = count;
    return (double)class_count/(double)count;
  }

  /*
   * Returns the projection metric (class label for discrete_projection / mean for numeric_projection)
   */
  double get_projection_metric() {
    return class_label;
  }
};

#endif
