#include "helper.hpp"
#include <algorithm>

struct object {
  double val;
  int index;
};

bool cmp(const object & s1, const object& s2)
{
   return s1.val < s2.val;
}

std::vector<int> *sort_indexes(const std::vector<double> &v) {
  std::vector<object> idx(v.size());
  for(unsigned int i=0; i<v.size(); i++) {
    idx[i].val = v[i];
    idx[i].index = i;
  }

  std::sort(idx.begin(), idx.end(), cmp);
  
  std::vector<int> *indices = new std::vector<int>(v.size());
  for(unsigned int i=0; i<v.size(); i++) 
    indices->at(i) = idx[i].index;

  return indices;
}

void vector_copy(std::vector<int> &a, std::vector<int> &b) {
  unsigned int size = a.size();
  b.resize(size);
  for(unsigned int i=0; i<size; i++)
    b[i] = a[i];
}

/*
 * Sort all dataset attributes except for label.
 * Store indices of sorted values
 */
indices_array *Helper::mk_indices_array_sorted_values(Datset &ds, std::vector<int> &train_rows) {
  int i, k, atts, rows;

  atts = ds.get_cols();
  rows = train_rows.size();
  
  indices_array *ia = new indices_array(atts);

  std::vector<double> d(rows);

  for(i=0; i<atts; i++) {
    for(k=0; k<rows; k++) {
      int row = train_rows[k];
      double value = ds.ds_real_ref(row, i);
      d[k] = value;
    }
    std::vector<int> *iv = sort_indexes(d);
    ia->set_indices(i, iv);
  }

  return ia;
}

void Helper::sort_indices_based_on_values(Datset &ds, int att, std::vector<int> &iv) {
  std::vector<object> idx(iv.size());
  for(unsigned int i=0; i<iv.size(); i++) {
    idx[i].val = ds.ds_real_ref(iv[i], att);
    idx[i].index = iv[i];
  }

  std::sort(idx.begin(), idx.end(), cmp);

  for(unsigned int i=0; i<iv.size(); i++) {
    iv[i] = idx[i].index;
  }
}

std::vector<int> *Helper::get_vector_subset(std::vector<int> &v, int start, int end) {
  std::vector<int>::const_iterator first = v.begin() + start;
  std::vector<int>::const_iterator last = v.begin() + end;
  std::vector<int> *newvec = new std::vector<int>(first, last);
  return newvec;
}

std::vector<int> *Helper::intersection(std::vector<int> &v1, std::vector<int> &v2)
{
  std::vector<int> *v3 = new std::vector<int>();
  
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  
  std::set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(), back_inserter(*v3));
  
  return v3;
}
