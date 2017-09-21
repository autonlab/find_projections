/*
   File:        search.cpp
   Author(s):   Saswati Ray
   Created:     Thu May  5 16:41:24 EDT 2016
   Description: 
   Copyright (c) 2016 Carnegie Mellon University
*/

#ifdef USE_PTHREADS
#include <pthread.h>
#endif /* USE_PTHREADS */

#include "discrete_binary_tree.hpp"
#include "numeric_binary_tree.hpp"
#include "helper.hpp"
#include "search.hpp"
#include "feature_map.hpp"
#include "feature_tree.hpp"
#include "discrete_projection.hpp"
#include "numeric_projection.hpp"

#include <vector>
#include <set>

static indices_array *remove_projection(indices_array &ia, projection *pr);

static bool validate_params(Datset &ds, int bin_size,int support, double purity_threshold, int num_threads, int mode) {
  printf("binsize = %d, support =  %d, purity =  %f, num_threads = %d, mode = %d\n", bin_size, support, purity_threshold, num_threads, mode);
  
  if(ds.is_valid() == false) {
    printf("Invalid dataset / label column\n");
    return false;
  }

  int size = ds.get_rows();
  if(bin_size <= 0 || bin_size > size) {
    printf("binsize should be a positive integer and less than your data size!\n");
    return false;
  }
  if(support <= 0 || support > size) {
    printf("support should be a positive integer and less than your data size!\n");
    return false;
  }

  if(purity_threshold <= 0.0 || purity_threshold > 1.0) {
    printf("purity should be a double between 0.0 - 1.0\n");
    return false;
  }

  return true;
}

static bool is_power_of_two(int n) {
  return !(n & (n-1));
}

static double compute_mean(Datset &ds, std::vector<int> &train_rows) {
  double truemean = 0.0;
  for(unsigned int i=0; i<train_rows.size(); i++) {
    int row = train_rows[i];
    double value = ds.ds_output_ref(row);
    truemean += value;
  }
  truemean /= train_rows.size();
 
  return truemean;
}

/*
 * Reset class distribution vectors at the leaf nodes
 */
static void reset_leaves_vector(std::vector<btree_node *> & leaves) {
  unsigned int i=0, gsize = leaves.size();
  for(i=0; i<gsize; i++) {
    btree_node *node = leaves[i];
    node->reset_node(false);
  }
}

static void process_projection_from_tree(std::vector<projection *> & pr_array, btree_node *node, Datset &ds, std::vector<int> &train_rows,
                     int i, std::vector<int> &ivatt1,
                     int j, std::vector<int> &ivatt2,
                     int m, int n,
                     int class_label, int pos, int neg) {
  projection *pr = projection::mk_projection_from_tree(node, ds, train_rows, ivatt1, m, n,
                               ivatt2, i, j);

  if(class_label >= 0) { // Discrete projection
    discrete_projection *dp = (discrete_projection *)pr;
    dp->set_class(class_label);
    dp->set_pos(pos);
    dp->set_neg(neg);
  }
  else { // Numeric projection
    numeric_projection *dp = (numeric_projection *)pr;
    numeric_binary_tree *nbt = (numeric_binary_tree *)node;
    dp->set_sum_sq_error(nbt->get_node_optimal_sum());
    dp->set_total(nbt->get_optimal_total());
    dp->set_mean(nbt->get_mean());
  }
  
  int gs, gensize = pr_array.size();
  bool overlap = false;
  bool better = false;
  /* Check if this projection overlaps with existing ones 
   * If there is overlap, which is the better one? */
  for(gs=0; gs<gensize && !overlap; gs++) {
    projection *qr = pr_array[gs];
    if(pr->do_projections_overlap(qr)) {
      overlap = true;
      if(qr->is_projection_better(pr)) {
        // Swap out the projection for the better one
        pr_array[gs] = pr;
        delete qr;
        better = true;
      }
    }
  }
  
  if(!overlap) {
    pr_array.push_back(pr);
  }
  else {
    if(!better)
      delete pr;
  }
}

/*
 * Function to create valid row subsets for feature 'i'.
 * Duplicates are put in same bin
 */
static std::vector<indices> *mk_feature_indices(std::vector<int> &ivatt1, Datset &ds, std::vector<int> &train_rows, int bin_size, int i) {
  int rows = ivatt1.size();
  std::vector<indices> *vec = new std::vector<indices>();

  int size = rows/bin_size;
  vec->reserve(size * size);

  /* Loop through all possible contiguous block of rows
   * Here rows mean the sorted values for a single dimension f1att
   */
  for(int m=0; m<rows; m+=bin_size) {
    double start_value = ds.ds_real_ref(train_rows[ivatt1[m]], i);
    double last_bin_value = -1;
    if(m-1 >= 0)
      last_bin_value = ds.ds_real_ref(train_rows[ivatt1[m-1]], i);
    while(m-1 >= 0 && m < rows-1) {
      if(start_value - last_bin_value < MIN_DOUBLE) {
        m++;
        start_value = ds.ds_real_ref(train_rows[ivatt1[m]], i);
      }
      else
        break;
    }
    
    for(int n=m+bin_size; n<rows; n+=bin_size) {
      double end_value;
      int ns;
      
      if(n >= rows)
        n = rows-1;
      end_value = ds.ds_real_ref(train_rows[ivatt1[n]], i);
      
      if(end_value - start_value < MIN_DOUBLE)
        continue;
      
      /* Try to advance index for duplicates/very very close values */
      for(ns=n+1; ns<rows; ns++) {
        double value = ds.ds_real_ref(train_rows[ivatt1[ns]], i);
        if(value - end_value < MIN_DOUBLE)
          n++;
        else
          break;
      }
      
      vec->push_back(indices(m, n));
    }
  }

  return vec;
}

/*
 * Function to evaluate any pair of dimensions (i,j)
 * Returns all projection-boxes (non-overlapping) for all classes which meet the criteria
 * support = Min. no. of data points to be contained in a projection-box
 * purity_threshold = Min purity of each projection found
 */
static std::vector<projection *> evaluate_attribute_pair(std::vector<int> &ivatt1, std::vector<int> &ivatt2, feature_tree *ftree, Datset &ds, std::vector<int> &train_rows, int bin_size,
                             int i, int j, int support, double purity_threshold, int exclude_class, int tree_mode) {
  /* For tracking best boxes for this projection */
  std::vector<projection *> pr_array;

  pr_array.reserve(10);

  bool is_classifier = ds.is_classification();

  /* Get binary tree for 'j' dimension */
  btree_node *node = ftree->getTree(j);
  std::vector<btree_node *> *leaves = ftree->getLeaves(j);

  /* Loop through all possible contiguous block of rows
   * Here rows mean the sorted values for a single dimension 'i'
   */
  std::vector<indices> *vec = ftree->getIndices(i);
  int orig_m = -1;
  int k = 0;
  for(unsigned int gs = 0; gs < vec->size(); gs++) {
    indices & index = vec->at(gs);

    int m = index.getm();
    if(m != orig_m) {
      orig_m = m;
      if(gs > 0) {
        k = 0;
        reset_leaves_vector(*leaves);
      }
    }
    
    int n = index.getn();
    int size = n-m+1;
    if(size < support) 
      continue;

    /* Add all the values for dimension 'j' incrementally into the tree */
    for(; k<size; k++) {
      int row = train_rows[ivatt1[k+m]];
      double value = ds.ds_real_ref(row, j);
      double score = ds.ds_output_ref(row);
      btree_node::insert_in_leaf_node(*leaves, value, score);
    } 

    if(is_classifier) {
      /* Evaluate best box for each class - Make it +ve, everything else -ve */
      int arity = ds.get_num_classes();
      for(int l=0; l<arity; l++) {
        if(l == exclude_class)
          continue;
        int pos=0, neg=0;
        
        discrete_binary_tree::set_leaves_pos(*leaves, l);
        
        node->update_tree();
        
        discrete_binary_tree *dbt = (discrete_binary_tree *)node;
        pos = dbt->get_opt_pos();
        neg = dbt->get_opt_neg();
        
        double purity = 0.0;
        if(pos + neg > 0)
          purity = (double)pos/(double)(pos+neg);
        bool match_box = ((pos+neg >= support) && (purity >= purity_threshold));
        /* Box found meeting selection criteria */
        if(match_box) 
          process_projection_from_tree(pr_array, node, ds, train_rows, i, ivatt1, j, ivatt2, m, n,
                                       l, pos, neg);
        
        node->reset_node(true);
      } /* Ends for loop for 'l' class */
    }
    else {
      node->update_tree();
      
      numeric_binary_tree *nbt = (numeric_binary_tree *)node;
      int total = nbt->get_optimal_total();
      bool match_box = total >= support;
      bool mean_proper = true;

      if(tree_mode == 1)
        mean_proper = nbt->get_mean() > purity_threshold;
      else if(tree_mode == 2)
        mean_proper = nbt->get_mean() < purity_threshold;

      /* Box found meeting selection criteria */
      if(match_box && mean_proper) 
        process_projection_from_tree(pr_array, node, ds, train_rows, i, ivatt1, j, ivatt2, m, n, -1, -1, -1);
      
      node->reset_node(true);
    } /* Ends else block */
  } /* Ends for loop for 'gs' */

  reset_leaves_vector(*leaves);

  return pr_array;
}

static pthread_mutex_t attribute_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct thread_struct {
  Datset *ds;
  indices_array *ia;
  int support;
  int bin_size;
  double purity;
  std::vector<int> atts_used;
  std::vector<int> train_rows;
  int mode;
  feature_map *table;
  feature_tree *ftree;
}thread_struct;

void *thread_routine (void *arg) {
  thread_struct *ts = (thread_struct *)arg;
  Datset *ds = ts->ds;
  indices_array *ia = ts->ia;
  feature_map *table = ts->table;
  int bin_size = ts->bin_size;
  std::vector<int> &atts_used = ts->atts_used;
  std::vector<int> &train_rows = ts->train_rows;
  int i, j, support = ts->support;
  double purity = ts->purity;
  int atts = ds->get_cols();
  int mode = ts->mode;
  feature_tree *ftree = ts->ftree;

  if(ds->is_classification() == false)
    purity = compute_mean(*ds, train_rows);

  /* Loop through all pairs of attributes */
  for(j=atts-1; j>0; j--) {
    // Synchronized block - Pick an unused attribute
    pthread_mutex_lock(&attribute_mutex);
    while(j > 0 && atts_used[j] != 0) {
      j--;
    }
    if(j > 0)
      atts_used[j] = 1;
    pthread_mutex_unlock(&attribute_mutex);

    if(j == 0)
      break;

    int f2att = j; /* X-axis */
    std::vector<int> &ivatt2 = ia->get_indices(f2att);

    for(i=j-1; i>=0; i--) {
      int f1att = i; /* Y-axis */
      std::vector<int> &ivatt1 = ia->get_indices(f1att);

      /* For tracking best boxes for this projection */
      std::vector<projection *> pr_array = evaluate_attribute_pair(ivatt1, ivatt2, ftree,
                                   *ds, train_rows, bin_size, i, j, support, purity, -1, mode);

      prlist prl(pr_array);
      table->setProjections(i, j, prl);
    }

    printf("Finished all projections containing feature %d\n", j);
  }
  
  return NULL;
}

static feature_tree *create_feature_tree(Datset &ds, indices_array &ia, std::vector<int> &train_rows, int bin_size, int tree_mode) {
  int atts = ds.get_cols();

  feature_tree *ftree = new feature_tree(atts);
  bool is_classifier = ds.is_classification();

  /* Loop through all attributes */
  for(int i=0; i<atts; i++) {
    std::vector<btree_node *> *leaves = new std::vector<btree_node *>();
    int f2att = i; /* X-axis */
    std::vector<int> &ivatt2 = ia.get_indices(f2att);

    /* Construct binary tree for 'f2att' dimension */
    btree_node *node = btree_node::construct_empty_tree(ds, i, is_classifier, train_rows, ivatt2, bin_size, leaves);
    ftree->setTree(i, node);

    if(!is_classifier) {
      numeric_binary_tree *nbt = (numeric_binary_tree *)node;
      if(tree_mode >=0 && tree_mode <= 2)
        nbt->set_mode(tree_mode);
    }
    
    ftree->setLeaves(i, leaves);

    std::vector<indices> *vec = mk_feature_indices(ivatt2, ds, train_rows, bin_size, i);
    ftree->setIndices(i, vec);
  }

  return ftree;
}

/*
 * Main function to exhaustively search for high-sum boxes for all 2-D projections.
 * num_threads :  No. of threads to use
 * bin_size : Size of data points in each tree leaf
 * support = Min. no. of data points to be contained in a projection-box
 * purity_threshold = Min purity of each projection found
 */
feature_map *search_for_max_subrectangles_threaded(Datset &ds, feature_tree *ftree, std::vector<int> &train_rows, int bin_size,
                           int support, double purity_threshold, int num_threads, int mode, indices_array &ia) {
  int i, atts;

  atts = ds.get_cols();

  feature_map *table = new feature_map(atts);

  std::vector<pthread_t> thread_id(num_threads);

  thread_struct args;
  args.ds = &ds;
  args.ia = &ia;
  args.bin_size = bin_size;
  args.table = table;
  args.train_rows = train_rows;
  args.support = support;
  args.purity = purity_threshold;
  args.mode = mode;
  args.ftree = ftree;

  std::vector<int> atts_used(atts);
  args.atts_used = atts_used;

  for(i=0; i<num_threads; i++) {
    pthread_create(&thread_id[i], NULL, thread_routine, &args);
  }

  for (i=0; i<num_threads; i++) {
    void *thread_result;
    pthread_join(thread_id[i], &thread_result);
  }

  return table;
}

/*
 * Main function to exhaustively search for high-sum boxes for all 2-D projections.
 * bin_size : Size of data points in each tree leaf
 * support = Min. no. of data points to be contained in a projection-box
 * purity_threshold = Min purity of each projection found
 */
feature_map *search_for_max_subrectangles(Datset &ds, feature_tree *ftree, std::vector<int> &train_rows,
                      int bin_size, int support, double purity_threshold, int mode, indices_array &ia) {
  int atts = ds.get_cols();

  feature_map *table = new feature_map(atts);

  if(ds.is_classification() == false)
    purity_threshold = compute_mean(ds, train_rows);

  /* Loop through all pairs of attributes */
  for(int i=0; i<atts-1; i++) {
    int f1att = i; /* Y-axis */
    std::vector<int> &ivatt1 = ia.get_indices(f1att);
    
    for(int j=i+1; j<atts; j++) {
      int f2att = j; /* X-axis */
      std::vector<int> &ivatt2 = ia.get_indices(f2att);
      
      /* For tracking best boxes for this projection */
      std::vector<projection *> array = evaluate_attribute_pair(ivatt1, ivatt2, ftree,
                                ds, train_rows, bin_size, i, j, support, purity_threshold, -1, mode);
      prlist prl(array);
      table->setProjections(i, j, prl);
    } /* Ends for loop for 'j' att */

    if(is_power_of_two(i))
      printf("Finished all projections containing feature number %d\n", i);
  } /* Ends for loop for 'i' att*/

  return table;
}

/*
 * Main function to exhaustively search for high-sum boxes for all 2-D projections.
 * bin_size : Size of data points in each tree leaf
 * num_threads :  No. of threads to use
 * support = Min. no. of data points to be contained in a projection-box
 * purity_threshold = Min purity of each projection found
 */
feature_map *search::search_projections(Datset& ds, int bin_size, int support, double purity_threshold, int mode, int num_threads) {
  feature_map *table = NULL;
  const clock_t begin_time = std::clock();
  bool valid = validate_params(ds, bin_size, support, purity_threshold, num_threads, mode);
  if(!valid)
    return NULL;

  int rows = ds.get_rows();
  std::vector<int> train_rows(rows);
  for(int i=0; i<rows; i++)
    train_rows[i] = i;

  /* Get sorted indices for all attributes.
   * This is done only once /
   */
  indices_array *ia = Helper::mk_indices_array_sorted_values(ds, train_rows);
  feature_tree *ftree = create_feature_tree(ds, *ia, train_rows, bin_size, mode);

  if(num_threads < 2)
    table = search_for_max_subrectangles(ds, ftree, train_rows, bin_size, support, purity_threshold, mode, *ia);
  else
    table = search_for_max_subrectangles_threaded(ds, ftree, train_rows, bin_size, support, purity_threshold, num_threads, mode, *ia);

  delete ftree;
  delete ia;

  printf("Time taken = %d sec\n", (int)(std::clock() - begin_time ) /  CLOCKS_PER_SEC);
  return table;
}

bool is_projection_better(projection *pr, int mode, bool is_numeric_problem, int maxsupport, double mean, double sqerr) {
  if(is_numeric_problem) {
    numeric_projection *np = (numeric_projection *)pr;
    switch(mode) {
    case 0:
      return (np->get_sum_sq_error() < sqerr);
    case 1:
      return (np->get_mean() > mean);
    case 2:
      return (np->get_mean() < mean);
    }
  }

  discrete_projection *dp = (discrete_projection *)pr;
  return (dp->get_total() > maxsupport);
}

/*
 * Learn decision list showing easily separable data
 */
projection_array *search::find_easy_explain_data(Datset& ds, double val_prop, int bin_size, int support, double purity, int mode,
                                                 int num_threads) {
  bool valid = validate_params(ds, bin_size, support, purity, num_threads, mode);
  if(!valid)
    return NULL;

  int i, j, k, rows = ds.get_rows();
  int atts = ds.get_cols();
  unsigned int tcount = 0;
  std::vector<int> seq(rows);

   for(int i=0; i<rows; i++)
    seq[i] = i;

  if(val_prop <= 0.0 || val_prop >= 1.0) {
    val_prop = 0.1;
    printf("Validation set proportion needs to be between 0 - 1\n");
  }

  int train_prop = rows - (int)(val_prop * rows + 0.5);
  std::random_shuffle(seq.begin(), seq.end());

  std::vector<int> *train_rows = Helper::get_vector_subset(seq, 0, train_prop);
  std::vector<int> *val_rows = Helper::get_vector_subset(seq, train_prop, rows);

  printf("No. of train rows = %lu, val rows = %lu\n", train_rows->size(), val_rows->size());

  projection *pr = NULL;
  indices_array *ia = Helper::mk_indices_array_sorted_values(ds, *train_rows);
  std::vector<double> proportions;
  std::vector<projection *> pr_array;

  bool is_numeric_problem = !(ds.is_classification());

  // Loop through all projections finding the best one greedily at each iteration
  // Stop when you can't find a projection meeting criteria
  do {
    feature_map *table = NULL;
    feature_tree *ftree = create_feature_tree(ds, *ia, *train_rows, bin_size, mode);

    if(num_threads <= 2)
      table = search_for_max_subrectangles(ds, ftree, *train_rows, bin_size, support, purity, mode, *ia);
    else
      table = search_for_max_subrectangles_threaded(ds, ftree, *train_rows, bin_size, support, purity, num_threads, mode, *ia);

    int maxsupport = 0;
    double sqerr = 1E6;
    double mean = (mode == 1) ? 0 : 1E6;

    if(pr) {
      delete pr;
      pr = NULL;
    }

    // Loop through all projections finding the best one greedily
    for(i=0; i<atts; i++) {
      for(j=i+1; j<atts; j++) {
        prlist array = table->getListOfProjections(i, j);
        for(k=0; k<array.size(); k++) {
          projection *bestprojection = array.get(k); 
          int valsupport = 0;
          if(is_projection_better(bestprojection, mode, is_numeric_problem, maxsupport, mean, sqerr) &&
             bestprojection->is_projection_good_on_validation_set(ds, *val_rows, mode, mean, purity, &valsupport)) {
            if(pr)
              delete pr;
            if(is_numeric_problem) {
              numeric_projection *np = new numeric_projection();
              mean = np->get_mean();
              sqerr = np->get_sum_sq_error();
              pr = np;
            }
            else {
              discrete_projection *dp = new discrete_projection();
              maxsupport = dp->get_total();
              pr = dp;
            }
            bestprojection->copy_projection(pr);  
          }
        } //End loop for k
      } // End loop for j
    } // End loop for i

    if(pr) { //Found best projection
      pr->pprojection();
      pr->mk_projection_indices(ds, *train_rows, *ia);

      projection *dp = NULL;
      if(is_numeric_problem) {
        dp = new numeric_projection();
        numeric_projection *np = (numeric_projection *)pr;
        double r2 = np->compute_R2(ds, *train_rows);
        printf("R2 = %f\n", r2);
      }
      else
        dp = new discrete_projection();

      indices_array *newia = remove_projection(*ia, pr);
      delete ia;
      ia = newia;
      tcount += pr->get_total();
      proportions.push_back((double)tcount/(double)train_rows->size());
   
      pr->copy_projection(dp);
      pr_array.push_back(dp);
    }

    delete table;
    delete ftree;
  } while(pr && tcount < train_rows->size());

  printf("easy data explained = %u / %lu (%f)\n\n", tcount, train_rows->size(), (double)tcount/(double)(train_rows->size()));

  projection::print_decision_list(pr_array, ds, proportions);

  projection_array *prarray = new projection_array(pr_array);

  delete train_rows;
  delete val_rows;
  delete ia;

  return prarray;
}

/*
 * Remove rows in projection from each indices in indices_array.
 */
indices_array *remove_projection(indices_array &ia, projection *pr) {
  int i, j;
  if(!pr)
    return NULL;

  std::vector<int> &common = *(pr->get_indices());
  std::set<int> ht;

  for(unsigned i=0; i<common.size(); i++) {
    int val = common[i];
    ht.insert(val);
  }

  int rows = ia.size();
  int cols = ia.get_indices(0).size() - common.size();
  indices_array *newia = new indices_array(rows);

  for(i=0; i<rows; i++) {
    std::vector<int> &iv = ia.get_indices(i);
    std::vector<int> *vec = new std::vector<int>(cols);
    newia->set_indices(i, vec);
    int k = cols-1;
    for(j=iv.size()-1; j>=0; j--) {
      int val = iv[j];
      const bool is_in = ht.find(val) != ht.end();
      if(is_in == false)  
        vec->at(k--) = val;
    }
  }
  
  return newia;
}

/*
 * Keep the nuggets in the projection as indicated by class.
 * Remove pure projections of opposite class within 'found_pr' projection'.
 */
void search::find_nuggets_in_projection(Datset& ds, std::vector<int> &train_rows, projection *found_pr, int bin_size) {
  int i, j, atts;
  projection *bestprojection = NULL;
  discrete_projection *bfound = (discrete_projection *)found_pr;
  int pos = bfound->get_pos();
  int neg = bfound->get_neg();
 
  if(ds.is_classification() == false) {
    printf("Please run this option for symbolic/discrete output only!\n");
    return;
  }

  atts = ds.get_cols();

  std::vector<int> &common = *(found_pr->get_indices());
  indices_array *ia = new indices_array(common.size());

  for(i=0; i<atts; i++) {
    ia->get_indices(i) = common;
    std::vector<int> &iv = ia->get_indices(i);
    int att = i;
    Helper::sort_indices_based_on_values(ds, att, iv);
  }

  printf("Trying to clean projection now\n");
  while(neg > 0) {
    int maxsum = 0;
    feature_tree *ftree = create_feature_tree(ds, *ia, train_rows, bin_size, -1);

    /* Loop through all pairs of attributes */
    for(i=0; i<atts-1; i++) {     
      int f1att = i; /* Y-axis */
      std::vector<int> &ivatt1 = ia->get_indices(f1att);

      for(j=i+1; j<atts; j++) {
    
        int f2att = j; /* X-axis */
        std::vector<int> &ivatt2 = ia->get_indices(f2att);
        
        /* For tracking best box for this projection */
        std::vector<projection *> garr = evaluate_attribute_pair(ivatt1, ivatt2, ftree,
                                                                 ds, train_rows, bin_size, i, j, 2, 1.0, bfound->get_class(), -1);
        for(unsigned int k=0; k<garr.size(); k++) {
          discrete_projection *pr = (discrete_projection *)garr[k];
          int this_sum = pr->get_pos();
          if(this_sum > maxsum) {
            maxsum = this_sum;
            if(bestprojection != NULL) 
              delete bestprojection;
            bestprojection = new discrete_projection();
            pr->copy_projection(bestprojection);
          }
        }
      } /* Ends for loop for 'j' att */
    } /* Ends for loop for 'i' att*/

    if(!bestprojection)
      break;
    
    bestprojection->pprojection();
    discrete_projection *bp = (discrete_projection *)bestprojection;
    neg -= bp->get_pos();
    printf("Left with %d pos and %d neg\n", pos, neg);

    bestprojection->mk_projection_indices(ds, train_rows, *ia);
    indices_array *newia = remove_projection(*ia, bestprojection);
    delete bestprojection;
    bestprojection = NULL;
    delete ia;
    ia = newia;
    delete ftree;
  } /* Ends while loop */

  delete ia;
  printf("\n\n");
}

/*
 * Keep the nuggets in the projection as indicated by class.
 * Remove pure projections of opposite class within 'found_pr' projection'.
 */
void search::find_class_nuggets(Datset& ds, int bin_size, int support, double purity) {
  projection *pr = NULL;
  int rows = ds.get_rows();
  int atts = ds.get_cols();

  if(ds.is_classification() == false) {
    printf("Please run this option for symbolic/discrete output only!\n");
    return;
  }

  std::vector<int> train_rows(rows);
  for(int i=0; i<rows; i++)
    train_rows[i] = i;

  indices_array *ia = Helper::mk_indices_array_sorted_values(ds, train_rows);

  do {
    feature_tree *ftree = create_feature_tree(ds, *ia, train_rows, bin_size, -1);
    feature_map *table = search_for_max_subrectangles(ds, ftree, train_rows, bin_size, support, purity, -1, *ia);

    int maxsupport = 0;
    if(pr) {
      delete pr;
      pr = NULL;
    }

    for(int i=0; i<atts; i++) {
      for(int j=i+1; j<atts; j++) {
        prlist & array = table->getListOfProjections(i, j);
        for(int k=0; k<array.size(); k++) {
          projection *bestprojection = array.get(k);
          if(bestprojection->get_total() > maxsupport) {
            if(pr)
              delete pr;
            pr = new discrete_projection();
            bestprojection->copy_projection(pr);
            maxsupport = bestprojection->get_total();
          }
        }
      }
    }

    if(pr) {
      pr->pprojection();
      pr->mk_projection_indices(ds, train_rows, *ia);
      delete ia;
      indices_array *newia = remove_projection(*ia, pr);
      ia = newia;
    }

    delete ia;
    delete table;
    delete ftree;
  } while(pr);
}
