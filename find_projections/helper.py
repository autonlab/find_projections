#!/bin/env python

##
# File:        helper.py
# Author(s):   Saswati Ray
# Created:     Thu May  5 16:41:24 EDT 2016
# Description: Python API wrapper
# Copyright (c) 2018 Carnegie Mellon University
##

import libfind_projections
from . import feature_map, datset
import numpy as np
import typing, sys, os
import pandas as pd

from d3m import container, utils
import d3m.metadata
from d3m.metadata.base import PrimitiveFamily
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params

def _add_target_semantic_types(metadata) -> metadata_base.DataMetadata:
    for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
        metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target')
        metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/TrueTarget')

    return metadata

def find_optimal_coverage(obj, ds, idf, odf, primitive, name) -> int:
    rows = ds.getSize()
    rowset = [i for i in range(rows)]

    bootstraps = 2
    print(primitive)
    primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    custom_hyperparams = dict()
    custom_hyperparams['n_estimators'] = 100

    baseline_accuracies = []
    global_dlist_accuracies = []
    global_dlist_coverages = []
    maxlen = 0
         
    from sklearn import metrics
    from scipy import stats
    import random

    prim_instance = primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))
    idfnew = pd.DataFrame(data=idf.values, columns=idf.columns.values.tolist())

    # Do bootstrap experiments
    for b in range(bootstraps):
        random.shuffle(rowset)
        train_ids = rowset[0:int(rows*0.8)]
        validation_ids = rowset[int(rows*0.8):]
         
        ds.ds.set_training_rows(np.ascontiguousarray(train_ids, dtype=float))
        fmap = obj.find_easy_explain_data()

        inputs = container.DataFrame(idfnew.iloc[train_ids,:], generate_metadata=False)
        inputs.metadata = idf.metadata.clear(for_value=inputs, generate_metadata=True)
        outputs = odf.iloc[train_ids,:]
        outputs.metadata = odf.metadata
        testdata = container.DataFrame(idfnew.iloc[validation_ids,:], generate_metadata=False)
        testdata.metadata = idf.metadata.clear(for_value=testdata, generate_metadata=True)
        to = odf.iloc[validation_ids,:]

        prim_instance.set_training_data(inputs=inputs, outputs=outputs)
        prim_instance.fit()
        predictions = prim_instance.produce(inputs=testdata).value.values

        if name is 'CLASSIFICATION':
            baseline_accuracy = metrics.accuracy_score(to, predictions)
        else:
            baseline_accuracy = metrics.mean_squared_error(to, predictions)
        print(baseline_accuracy)
        baseline_accuracies.append(baseline_accuracy)

        num = fmap.get_num_projections()
        if num > maxlen:
            maxlen = num
        
        global_accuracies = []
        global_coverages = []
        for k in range(1,num+1):
            predictedTargets = np.zeros(len(validation_ids))
            coverage = 0
            for i in range(len(validation_ids)):
                row = validation_ids[i]
                predicted = False
                for j in range(k):
                    pr = fmap.get_projection(j)
                    if pr.point_lies_in_projection(ds.ds, row) is True:
                        predictedTargets[i] = pr.get_projection_metric()
                        predicted = True
                        coverage = coverage + 1
                        break
                if predicted == False:
                    predictedTargets[i] = predictions[i]
            coverage = (float)(coverage) / (float)(len(validation_ids))
            if name is 'CLASSIFICATION':
                hybrid_accuracy = metrics.accuracy_score(to, predictedTargets)
            else:
                hybrid_accuracy = metrics.mean_squared_error(to, predictedTargets)
            global_accuracies.append(hybrid_accuracy)
            global_coverages.append(coverage)
        global_dlist_accuracies.append(global_accuracies)
        global_dlist_coverages.append(global_coverages)
    # boostrapping done

    mean, var, std = stats.bayes_mvs(baseline_accuracies, 0.95)
    baseline_lb = mean[1][0]
         
    optimal_coverage = 0.0
    hybrid_acc = 0.0
    hybrid_accuracies = []
    hybrid_coverages = []
    for i in range(1,maxlen):
        gacc = []
        gcvg = []
        for b in range(bootstraps):
            acc = global_dlist_accuracies[b]
            cvg = global_dlist_coverages[b]
            if i < len(acc):
                gacc.append(acc[i])
                gcvg.append(cvg[i])
        
        print("len ", i)
        print("gacc ", gacc)
        print("gcvg ", gcvg)
        # Computing upper bounds
        if len(gacc) == 1:
            hybrid_accuracies.append(gacc[0])
            hybrid_coverages.append(gcvg[0])
        else:
            mean, var, std = stats.bayes_mvs(gacc, 0.95)
            hybrid_accuracies.append(mean[1][1])
            mean, var, std = stats.bayes_mvs(gcvg, 0.95)
            hybrid_coverages.append(mean[1][1])

    optimal_cvg = 0.0
    index = -1
    for acc in hybrid_accuracies:
        if name is 'CLASSIFICATION'and acc >= baseline_lb:
            index = index+1
        elif name is 'REGRESSION' and acc <= baseline_lb:
            index = index + 1
        else:
            break

    if index >= 0:
        optimal_cvg = hybrid_coverages[index]

    print("optimal: ", optimal_cvg)
    return optimal_cvg
