#!/bin/env python27

import sys, csv
import numpy
import find_projections.search_projections as search_projections

# Read input feature set
result = numpy.random.rand(1000,2).astype("float")

# CLASSIFICATION
# Read output feature for classification
output = numpy.random.randint(2, size=(1000,)).astype("float")

# Create search object and parameters
search_object = search_projections.Search(purity=0.6)

search_object.set_training_data(result, output)

# Search comprehensively for projection boxes
fmap = search_object.search_projections()

num = fmap.get_num_projections()
# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  #print pr.get_total()
  pr.pprojection()

# Search for easy-to-classify data (decision list)
fmap = search_object.find_easy_explain_data()

num = fmap.get_num_projections()
# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  #print pr.get_total()
  pr.pprojection()

# REGRESSION
p = search_object.get_params()
p = p._replace(is_classifier = False)
p = p._replace(support = 10)
search_object.set_params(p)
search_object.set_training_data(result, output)

# Search comprehensively for projection boxes
fmap = search_object.search_projections()

num = fmap.get_num_projections()

# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  #print pr.get_mean()
  pr.pprojection()
