#!/bin/env python

##
# File:        search_projections.py
# Author(s):   Saswati Ray
# Created:     Thu May  5 16:41:24 EDT 2016
# Description: Python API wrapper
# Copyright (c) 2016 Carnegie Mellon University
##

import os
from typing import Any

import d3m.metadata
import libfind_projections
import numpy as np
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params
from d3m.metadata.base import PrimitiveFamily
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitives.classification.random_forest import SKlearn as SKRandomForestClassifier
from d3m.primitives.classification.gradient_boosting import SKlearn as GradientBoostingClassifier
from sklearn import preprocessing

from . import feature_map, datset, helper

Input = container.DataFrame
Output = container.DataFrame


class SearchHybridParams(params.Params):
    is_fitted: bool
    prim_instance: Any
    fmap: Any
    fmap_py: Any
    default_value: Any
    le: Any
    num: Any


class SearchHybridHyperparams(hyperparams.Hyperparams):
    binsize = hyperparams.UniformInt(lower=1, upper=1000, default=10,
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                     description='No. of data points for binning each feature.')
    support = hyperparams.UniformInt(lower=1, upper=10000, default=100,
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                     description='Minimum number of data points to be present in a projection box for '
                                                 'evaluation.')
    purity = hyperparams.Uniform(lower=0.01, upper=1.0, default=0.9,
                                 semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                 description='Minimum purity (class proportion) in a projection box for discrete '
                                             'class output.')
    num_threads = hyperparams.UniformInt(lower=1, upper=10, default=1, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
                                         description='No. of threads for multi-threaded operation.')
    validation_size = hyperparams.Uniform(lower=0.01, upper=0.5, default=0.1, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                                          description='Proportion of training data which is held out for validation '
                                                      'purposes.')
    blackbox = hyperparams.Primitive[SupervisedLearnerPrimitiveBase](
        primitive_families=[PrimitiveFamily.CLASSIFICATION],
        default=GradientBoostingClassifier,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Black box model to fall back after decision list.')


class SearchHybrid(SupervisedLearnerPrimitiveBase[Input, Output, SearchHybridParams, SearchHybridHyperparams]):
    """
    Class to search for 2-d projection boxes in raw feature space for discrete(categorical) output (for
    classification problems) . For discrete output, the algorithm tries to find 2-d projection boxes which can
    separate out any class of data from the rest with high purity.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "id": "448590e7-8cf6-4bfd-abc4-db2980d8114e",
        "version": "2.3.1",
        "name": "find projections",
        "description": "Searching 2-dimensional projection boxes in raw data separating out homogeneous data points",
        "python_path": "d3m.primitives.classification.search_hybrid.Find_projections",
        "primitive_family": "CLASSIFICATION",
        "algorithm_types": ["ASSOCIATION_RULE_LEARNING", "DECISION_TREE"],
        "keywords": ["classification", "rule learning"],
        "source": {
            "name": "CMU",
            "contact": "mailto:sray@cs.cmu.edu",
            "uris": ["https://gitlab.datadrivendiscovery.org/sray/find_projections.git"]
        },
        "precondition": [metadata_base.PrimitivePrecondition.NO_CATEGORICAL_VALUES],
        "installation": [
            {
                "type": "UBUNTU",
                "package": "libboost-python1.65-dev",
                "version": "1.65.1"
            },
            {
                "type": "PIP",
                "package_uri": 'git+https://github.com/autonlab/find_projections.git@{git_commit}#egg=find_projections'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ]
    })

    def __init__(self, *, hyperparams: SearchHybridHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._search_obj = libfind_projections.search()
        self.hyperparams = hyperparams
        self._ds = None
        self._fmap = None
        self._fmap_py = None
        self._is_fitted = False
        self._default_value = None
        self._inputs = None
        self._outputs = None
        self._num = None
        self._prim_instance = None
        self._le = preprocessing.LabelEncoder()

    def __getstate__(self):
        return (
        self.hyperparams, self._fmap_py, self._default_value, self._num, self._prim_instance, self._is_fitted, self._le)

    def __setstate__(self, state):
        self.hyperparams, self._fmap_py, self._default_value, self._num, self._prim_instance, self._is_fitted, self._le = state
        self._fmap = None

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
        return feature_map.FeatureMap(
            self._search_obj.search_projections(self._ds.ds, self.hyperparams['binsize'], self.hyperparams['support'],
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
        return feature_map.FeatureMap(
            self._search_obj.find_easy_explain_data(self._ds.ds, self.hyperparams['validation_size'],
                                                    self.hyperparams['binsize'],
                                                    self.hyperparams['support'], self.hyperparams['purity'], 1,
                                                    self.hyperparams['num_threads']))

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

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        primitive = self.hyperparams['blackbox']
        idf = self._inputs
        odf = container.DataFrame(self._outputs, generate_metadata=True)
        optimal_cvg = helper.find_optimal_coverage(self, self._ds, idf, odf, primitive, 'CLASSIFICATION',
                                                   random_seed=self.random_seed)
        self._fmap = self.find_easy_explain_data()
        self._fmap_py = []

        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        custom_hyperparams = dict()
        custom_hyperparams['n_estimators'] = 100
        if isinstance(primitive, d3m.primitive_interfaces.base.PrimitiveBaseMeta):  # is a class
            self._prim_instance = primitive(
                hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))
        else:  # is an instance
            self._prim_instance = primitive
        self._prim_instance.set_training_data(inputs=idf, outputs=odf)
        self._prim_instance.fit()

        self._num = -1
        num = self._fmap.get_num_projections()
        for i in range(num):
            pr = self._fmap.get_projection(i)
            cvg = pr.get_coverage()

            if cvg > optimal_cvg:
                break

            self._num = i
            att1 = pr.get_att1()
            att2 = pr.get_att2()
            start1 = pr.get_att1_start()
            start2 = pr.get_att2_start()
            end1 = pr.get_att1_end()
            end2 = pr.get_att2_end()
            value = pr.get_projection_metric()
            self._fmap_py.append((att1, att2, start1, start2, end1, end2, value,))
        self._is_fitted = True
        return base.CallResult(None)

    """
     Sets input and output feature space.
     Parameters
     ----------
     inputs : Input
         A nxd DataFrame of training data points (dense, no missing values)

     outputs: Output
         A nx1 DataFrame of floats (dense)

     """

    def set_training_data(self, *, inputs: Input, outputs: Output) -> None:
        self._ds = datset.Datset(np.ascontiguousarray(inputs.values, dtype=float))
        v = self._le.fit_transform(outputs.values.ravel())
        self._ds.setOutputForClassification(np.ascontiguousarray(v, dtype=float))

        self._inputs = inputs
        self._outputs = v

        self._fmap = None
        self._fmap_py = None
        self._is_fitted = False
        self._default_value = self._ds.get_default_value()

    """
     Returns all the search parameters in Params object
     """

    def get_params(self) -> SearchHybridParams:
        return SearchHybridParams(is_fitted=self._is_fitted,
                                  prim_instance=self._prim_instance,
                                  fmap=self._fmap,
                                  fmap_py=self._fmap_py,
                                  default_value=self._default_value,
                                  le=self._le,
                                  num=self._num)

    """
     Sets all the search parameters from a Params object
     :param is_classifier: True for discrete-class output. False for numeric output.
     :type: boolean
     :type: Double
     """

    def set_params(self, *, params: SearchHybridParams) -> None:
        self._is_fitted = params['is_fitted']
        self._prim_instance = params['prim_instance']
        self._fmap = params['fmap']
        self._fmap_py = params['fmap_py']
        self._default_value =params['default_value']
        self._le = params['le']
        self._num = params['num']

    """
     Returns predictions made on test data from prior saved list of projections.
     Parameters
     ----------
     inputs : Input
         A nxd DataFrame of test data points

     Returns
     -------
     Predict
         A nx1 DataFrame of predictions

     """

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> base.CallResult[Output]:
        if self._fmap is None and self._fmap_py is None:
            # TODO `produce` should never return `None` but just abort if it cannot run; throw an exception or something
            return base.CallResult(None)

        testds = datset.Datset(np.ascontiguousarray(inputs.values, dtype=float))
        rows = testds.getSize()
        predictedTargets = np.zeros(rows, dtype=np.int8)

        clfp = self._prim_instance.produce(inputs=inputs).value.values

        # Loop through all the test rows
        for j in range(rows):

            # Loop through all the projections in order of attributes
            predicted = False
            if bool(self._fmap):
                num = self._num
                for i in range(num + 1):
                    pr = self._fmap.get_projection(i)
                    if pr.point_lies_in_projection(testds.ds, j) is True:
                        predictedTargets[j] = (int)(pr.get_projection_metric())
                        predicted = True
                        break
            else:
                (value, predicted) = testds._helper(self._num, self._fmap_py, j)
                if predicted is True:
                    predictedTargets[j] = (int)(value)

            # Predict using outside blackbox classifier
            if predicted is False:
                predictedTargets[j] = (int)(clfp[j])

        predictedTargetNames = self._le.inverse_transform(predictedTargets)
        output = container.DataFrame(predictedTargetNames, generate_metadata=True)
        return base.CallResult(output)
