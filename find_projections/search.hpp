/*
   File:        search.hpp
   Author(s):   Saswati Ray
   Created:     Fri Feb 19 16:23:48 EST 2016
   Description: 
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifndef SEARCH_H_
#define SEARCH_H_

#include "datset.hpp"
#include "projection.hpp"
#include "feature_map.hpp"

class search {
private:
public:

/*
 * Keep the nuggets in the projection as indicated by class.
 * Remove pure projections of opposite class within 'found_pr' projection'.
 */
  void find_nuggets_in_projection(Datset& ds, std::vector<int> &train_rows, projection *found_pr, int bin_size);

/*
 * Main function to exhaustively search for high-sum boxes for all 2-D projections.
 * bin_size : Size of data points in each tree leaf
 * num_threaDs :  No. of threaDs to use
 * support : Min. no. of data points to be contained in a projection-box
 * mode : Mode of operation for numeric output
 * 0 : Tries to find low variance boxes
 * 1 : Tries to find high mean boxes
 * 2 : Tries to find low mean boxes
 */
  feature_map *search_projections(Datset& Ds, int bin_size, int support, double purity_threshold, int mode, int num_threads);

/*
 * Learn decision list showing easily separable data
 */
  projection_array *find_easy_explain_data(Datset& Ds, double val_prop, int bin_size, int support, double purity_threshold, int mode,
                                           int num_threads);

  void find_class_nuggets(Datset& Ds, int bin_size, int support, double purity);
};

#endif
