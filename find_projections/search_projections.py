#!/bin/env python

##
# File:        search_projections.py
# Author(s):   Saswati Ray
# Created:     Thu May  5 16:41:24 EDT 2016
# Description: Python API wrapper
# Copyright (c) 2016 Carnegie Mellon University
##

import libfind_projections
import numpy as np
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from typing import NamedTuple

def validate_params(ds, binsize, support, purity, mode, num_threads):
    if ds.isValid() is False:
        return False

    size = ds.getSize()

    if ( binsize <= 0 )  or ( binsize >= size ):
        return False
        
    if ( support >= size ) :
        return False
        
    if ( purity <= 0.0 ) or ( purity >= 1.0 ) :
        return False  

    if ( num_threads < 1 ) :
        return False;

    return True

Input = np.ndarray
Output = np.ndarray
Params = NamedTuple('Params', [
    ('is_classifier', bool),
    ('binsize', int),
    ('support', int),
    ('purity', float),
    ('mode', int),
    ('num_threads', int),
    ('validation_size', float)    
])

class Search(SupervisedLearnerPrimitiveBase[Input, Output, Params]):
     """
     Class to perform different types of search operations
     :param is_classifier: True for discrete-class output. False for numeric output.
     :type: boolean
     :param binsize: No. of data points for binning. Should be a positive integer
     :type: Integer
     :param support: Minimum number of data points to be present in a projection box for evaluation. Should be a positive integer
     :type: Integer
     :param purity: Minimum purity (class proportion) in a projection box. Should be in the range 0.0 - 1.0
     :type: Double
     :param mode: Used for numeric output (regression-based analysis). Valid values are 0, 1, 2.
     :type: Integer
     :param num_threads: No. of threads for multi-threaded operation. Should be a positive integer
     :type: Integer
     :param validation_size: Proportion of training data which is held out for validation purposes. Should be in the range 0.0 - 0.5
     :type: Double
     """
     def __init__(self, is_classifier=True, binsize=10, support=100, purity=0.9, mode=1, num_threads=1, validation_size=0.1):
         super(Search, self).__init__()
         self.search_obj = libfind_projections.search()
         self.binsize = binsize
         self.support = support
         self.purity = purity
         self.mode = mode
         self.num_threads = num_threads
         self.validation_size = validation_size
         self.ds = None
         self.fmap = None
         self.is_classifier = True
         
     """
     Comprehensively evaluates all possible pairs of 2-d projections in the data
     Returns all projection boxes which match search criteria
     :returns: FeatureMap instance containing all the projection boxes found meeting the search criteria
     :rtype: FeatureMap
     """
     def search_projections(self):
         valid = validate_params(self.ds, self.binsize, self.support, self.purity, self.mode, self.num_threads)
         if valid is False:
             print("Invalid parameters!")
             return None
         return FeatureMap(self.search_obj.search_projections(self.ds.ds, self.binsize, self.support, self.purity, self.mode, self.num_threads))

     """
     Learns decision list of projection boxes for easy-to-explain data (for classification/regression)
     Returns projection boxes in a decision-list based scheme (if-else-if)
     :returns: FeatureMap instance containing all the projection boxes found meeting the search criteria
     :rtype: FeatureMap
     """
     def find_easy_explain_data(self):
         valid = validate_params(self.ds, self.binsize, self.support, self.purity, self.mode, self.num_threads)
         if valid is False:
             print("Invalid parameters!")
             return None
         if ( self.validation_size <= 0.0 ) or ( self.validation_size > 0.5 ) :
             return None
         return FeatureMap(self.search_obj.find_easy_explain_data(self.ds.ds, self.validation_size, self.binsize, self.support, self.purity, self.mode, self.num_threads))

     """
     Learns decision list of projection boxes for easy-to-explain data (for classification/regression)
     """
     def fit(self, timeout=None, iterations=None):
         self.fmap = self.find_easy_explain_data() 

     """
     Sets input and output feature space.
     :param inputs: 2-d numpy array of floats (dense, no missing values)
     :param outputs: 1-d numpy array of floats (dense)
     """
     def set_training_data(self, inputs, outputs):
         self.ds = Datset(np.ascontiguousarray(inputs, dtype=float))
         if self.is_classifier is True:
           self.ds.setOutputForClassification(np.ascontiguousarray(outputs, dtype=float))
         else:
           self.ds.setOutputForRegression(np.ascontiguousarray(outputs, dtype=float))

     """
     Returns all the search parameters in Params object
     """
     def get_params(self):
         return Params(is_classifier=self.is_classifier, binsize=self.binsize, support=self.support, purity=self.purity, mode=self.mode, num_threads=self.num_threads, validation_size=self.validation_size)

     """
     Sets all the search parameters from a Params object
     :param is_classifier: True for discrete-class output. False for numeric output.
     :type: boolean
     :param binsize: No. of data points for binning. Should be a positive integer
     :type: Integer
     :param support: Minimum number of data points to be present in a projection box for evaluation. Should be a positive integer
     :type: Integer
     :param purity: Minimum purity (class proportion) in a projection box. Should be in the range 0.0 - 1.0
     :type: Double
     :param mode: Used for numeric output (regression-based analysis). Valid values are 0, 1, 2.
     :type: Integer
     :param num_threads: No. of threads for multi-threaded operation. Should be a positive integer
     :type: Integer
     :param validation_size: Proportion of training data which is held out for validation purposes. Should be in the range 0.0 - 0.5
     :type: Double
     """
     def set_params(self, params):
         self.is_classifier=params.is_classifier
         self.binsize=params.binsize
         self.support=params.support
         self.purity=params.purity
         self.mode=params.mode
         self.num_threads=params.num_threads
         self.validation_size=params.validation_size

     """
     Returns predictions made on test data from prior saved list of projections.
     """
     def produce(self, inputs):
         if self.fmap is None:
             return None

         testds = Datset(np.ascontiguousarray(inputs, dtype=float))
         rows = testds.get_size()
         predictedTargets = numpy.zeros(rows)

         # Loop through all the test rows
         for j in range(rows):

             # Loop through all the projections in order of attributes
             predicted = false
             for i in range(num):
                 pr = self.fmap.get_projection(i)
                 if pr.point_lies_in_projection(testDs, j) is true:
                     predictedTargets[j] = pr.get_projection_metric()
                     predicted = true
                     break

                     # Predict using outside blackbox classifier
                     predictedTargets[j] = -1 #clf.predict(testData[j,:])

         return predictedTargets

class FeatureMap:
    """
    Container class containing projection boxes found from search operations
    """
    def __init__(self, fmap):
        self.fmap = fmap
        
    """
    Returns the total number of projection boxes in this container object
    """
    def get_num_projections(self):
        return self.fmap.get_num_projections()
    
    """
    Retrieve the i'th projection-box
    """
    def get_projection(self, i):
        if ( i <  0 ) :
            return None
        return self.fmap.get_projection(i)
    
class Datset:

     """
     Create Datset instance with numpy 2-d array of floats.
     """
     def __init__(self, data):
         rows = data.shape[0]
         cols = data.shape[1]

         if data.dtype not in np.sctypes['float']:
             return None

         self.ds = libfind_projections.Datset(data)
        
     """
     Set output array for classification task
     """ 
     def setOutputForClassification(self, output):
         if ( np.issubdtype(output.dtype, np.float ) ) :
            self.ds.fill_datset_output_for_classification(output)  
         else:
            raise Exception("Invalid classification data type")

     """
     Set output array for regression task
     """ 
     def setOutputForRegression(self, output):
         if ( np.issubdtype(output.dtype, np.float ) ) :
            self.ds.fill_datset_output_for_regression(output)  
         else:
            raise Exception("Invalid regressionion data type")

     """
     Checks if Datset instance has been populated properly.
     """
     def isValid(self):
         return self.ds.is_valid()

     """
     Returns the number of data points
     """
     def getSize(self):
         return self.ds.get_size()
