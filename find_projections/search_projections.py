#!/bin/env python

##
# File:        search_projections.py
# Author(s):   Saswati Ray
# Created:     Thu May  5 16:41:24 EDT 2016
# Description: Python API wrapper
# Copyright (c) 2016 Carnegie Mellon University
##

import libfind_projections
from . import feature_map, datset
import numpy as np
import typing

from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import d3m_metadata
from d3m_metadata.metadata import PrimitiveMetadata
from d3m_metadata import hyperparams
from d3m_metadata import params

Input = d3m_metadata.container.ndarray
Output = d3m_metadata.container.ndarray
Predict = d3m_metadata.container.ndarray

class SearchParams(params.Params):
     is_fitted: bool

class SearchHyperparams(hyperparams.Hyperparams):
     binsize = hyperparams.UniformInt(lower=1, upper=1000,default=10,description='No. of data points for binning each feature.')
     support = hyperparams.UniformInt(lower=1, upper=10000,default=100,description='Minimum number of data points to be present in a projection box for evaluation.')
     purity = hyperparams.Uniform(lower=0.01, upper=1.0,default=0.9,description='Minimum purity (class proportion) in a projection box for discrete class output.')
     num_threads = hyperparams.UniformInt(lower=1, upper=10,default=1,description='No. of threads for multi-threaded operation.')
     validation_size = hyperparams.Uniform(lower=0.01, upper=0.5,default=0.1,description='Proportion of training data which is held out for validation purposes.')

class Search(SupervisedLearnerPrimitiveBase[Input, Output, SearchParams, SearchHyperparams]):
     """
     Class to search for 2-d projection boxes in raw feature space for discrete(categorical) output (for classification problems) .
     For discrete output, the algorithm tries to find 2-d projection boxes which can separate out any class of data from the rest with high purity.
     """

     metadata = PrimitiveMetadata({
         "id": "84f39131-6618-4d90-9590-b79d41dfb093",
         "version": "2.0",
         "name": "find projections",
         "description": "Searching 2-dimensional projection boxes in raw data separating out homogeneous data points",
         "python_path": "d3m.primitives.cmu.autonlab.find_projections.Search",
         "primitive_family": "CLASSIFICATION",
         "algorithm_types": [ "ASSOCIATION_RULE_LEARNING", "DECISION_TREE" ],
         "keywords": ["classification", "rule learning"],
         "source": {
             "name": "CMU",
             "uris": [ "https://gitlab.datadrivendiscovery.org/sray/find_projections.git" ]
         },
         "installation": [
         {
             "type": "PIP",
             "package_uri": "git+https://gitlab.datadrivendiscovery.org/sray/find_projections.git"
         },
         {
             "type": "UBUNTU",
             "package": "libboost-all-dev",
             "version": "1.65.1"
         }
         ]
     })

     def __init__(self, *, hyperparams: SearchHyperparams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
         self._search_obj = libfind_projections.search()
         self.hyperparams = hyperparams
         self._ds = None
         self._fmap = None
         self._is_fitted = False
         self.random_seed = random_seed
         self.docker_containers = docker_containers
         
     """
     Comprehensively evaluates all possible pairs of 2-d projection boxes in the data
     Returns all projection boxes which match search criteria
     Returns
     -------
     FeatureMap instance containing all the projection boxes found
     """
     def search_projections(self) -> feature_map.FeatureMap:
         valid = datset.validate_params(self._ds, self.hyperparams['binsize'], self.hyperparams['support'])
         if valid is False:
             print("Invalid parameters!")
             return None
         return feature_map.FeatureMap(self._search_obj.search_projections(self._ds.ds, self.hyperparams['binsize'], self.hyperparams['support'],
          self.hyperparams['purity'], 1, self.hyperparams['num_threads']))

     """
     Learns decision list of projection boxes for easy-to-explain data (for classification)
     Returns projection boxes in a decision-list based scheme (if-else-if)
     Returns
     -------
     FeatureMap instance containing all the projection boxes found
     """
     def find_easy_explain_data(self) -> feature_map.FeatureMap:
         valid = datset.validate_params(self._ds, self.hyperparams['binsize'], self.hyperparams['support'])
         if valid is False:
             print("Invalid parameters!")
             return None
         return feature_map.FeatureMap(self._search_obj.find_easy_explain_data(self._ds.ds, self.hyperparams['validation_size'], self.hyperparams['binsize'],
          self.hyperparams['support'], self.hyperparams['purity'], 1, self.hyperparams['num_threads']))

     """
     Return the FeatureMap instance containing all the projection boxes learnt
     Returns
     -------
     FeatureMap instance containing all the projection boxes found
     """
     def get_feature_map(self) -> feature_map.FeatureMap:
         return self._fmap

     """
     Learns decision list of projection boxes for easy-to-explain data (for classification)
     """
     def fit(self, *, timeout: float = None, iterations: int = None) -> None:
         self._fmap = self.find_easy_explain_data() 

     """
     Sets input and output feature space.
     Parameters
     ----------
     inputs : Input
         A nxd matrix of training data points (dense, no missing values)

     outputs: Output
         A nx1 numpy array of floats (dense)

     """
     def set_training_data(self, *, inputs: Input, outputs: Output) -> None:
         self._ds = datset.Datset(np.ascontiguousarray(inputs, dtype=float))
         self._ds.setOutputForClassification(np.ascontiguousarray(outputs, dtype=float))
         
         self._fmap = None
         self._is_fitted = True

     """
     Returns all the search parameters in Params object
     """
     def get_params(self) -> SearchParams:
         return SearchParams(is_fitted = self._is_fitted)

     """
     Sets all the search parameters from a Params object
     :param is_classifier: True for discrete-class output. False for numeric output.
     :type: boolean
     :type: Double
     """
     def set_params(self, *, params: SearchParams) -> None:
         self._is_fitted = params.is_fitted

     """
     Returns predictions made on test data from prior saved list of projections.
     Parameters
     ----------
     inputs : Input
         A nxd matrix of test data points

     Returns
     -------
     Predict
         A nx1 array of predictions

     """
     def produce(self, *, inputs: Input) -> Predict:
         if self._fmap is None:
             return None

         testds = datset.Datset(np.ascontiguousarray(inputs, dtype=float))
         rows = testds.getSize()
         predictedTargets = np.zeros(rows)
         num = self._fmap.get_num_projections()

         # Loop through all the test rows
         for j in range(rows):

             # Loop through all the projections in order of attributes
             predicted = False
             for i in range(num):
                 pr = self._fmap.get_projection(i)
                 if pr.point_lies_in_projection(testds.ds, j) is True:
                     predictedTargets[j] = pr.get_projection_metric()
                     predicted = True
                     break

             # Predict using outside blackbox classifier
             if predicted is False:
               predictedTargets[j] = -1 #clf.predict(testData[j,:])

         return predictedTargets
