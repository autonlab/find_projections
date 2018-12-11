
import sys, csv
import numpy
from sklearn import preprocessing
import pandas as pd

#from d3m.primitives.cmu.autonlab.find_projections import Search, SearchNumeric

import find_projections.search_projections as search_projections
import find_projections.search_numeric_projections as search_numeric_projections

# Read input feature set
result = numpy.random.rand(1000,2).astype("float")
rf = pd.DataFrame(data=result, columns=['x', 'y'])

# CLASSIFICATION
# Read output feature for classification
output = numpy.random.randint(2, size=(1000,)).astype("float")
of = pd.DataFrame(data=output, columns=['Label'])

hyperparams = search_projections.SearchHyperparams(purity=0.75, binsize=10, support=25, num_threads=1, validation_size=0.1)
# Create search object and parameters
search_object = search_projections.Search(hyperparams=hyperparams)

search_object.set_training_data(inputs=rf, outputs=of)

# Search comprehensively for projection boxes
fmap = search_object.search_projections()

num = fmap.get_num_projections()
# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  pr.pprojection()

# Search for easy-to-classify data (decision list)
search_object.fit()

predicted = search_object.produce(inputs=rf).value

fmap = search_object.get_feature_map()
num = fmap.get_num_projections()
# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  #print pr.get_total()
  pr.pprojection()

# REGRESSION
hyperparams = search_numeric_projections.SearchNumericHyperparams(binsize=10, support=25, mode=1, num_threads=1, validation_size=0.1)
# Create search object and parameters
search_object = search_numeric_projections.SearchNumeric(hyperparams=hyperparams)
search_object.set_training_data(inputs=rf, outputs=of)

# Search or easy-to-regress data (decision list)
search_object.fit()
predicted = search_object.produce(inputs=rf).value

fmap = search_object.get_feature_map()
num = fmap.get_num_projections()

# Loop through all the projections in order of attributes
for i in range(num):
  pr = fmap.get_projection(i)
  #print pr.get_mean()
  pr.pprojection()
