import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

from weka.core import jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.filters import Filter

from copy import deepcopy
import sklearn

from classifier import train_trees, get_initial_weights,sigmoid,neuron_l1,train, test
from data_loader import load_data, make_partition

parser = argparse.ArgumentParser()
parser.add_argument('--training_data', dest = 'train_path')
parser.add_argument('--testing_data', dest = 'test_path')
parser.add_argument('--random_initialization', action= 'store_true')
parser.add_argument('--outfile', dest = 'outfile')

args = parser.parse_args()

if __name__ == '__main__':

    train_path = args.train_path
    test_path = args.test_path
    rand_init = args.random_initialization
    outfile = args.outfile

    jvm.start()

    data, attributes = load_data(train_path)
    data_normal, N = make_partition(data,attributes)

    print('\n\n############ Training Decision Trees ############\n\n')

    clfs,dt_y_hat = train_trees(data_normal,attributes)
    w2_init,w1_init,b1_init = get_initial_weights(data_normal,clfs,attributes,dt_y_hat)

    print('\n\nDone!')

    print('\n\n############ Training Hidden Layer Parameters ############\n\n')

    w1,b1=train(w1_init = w1_init,
                b1_init = b1_init,
                lr = 0.01,
                iterations = 100,
                N = N,
                data = data_normal,
                attributes = attributes,
                dt_y_hat = dt_y_hat)

    data_test,attributes_test = load_data(test_path)
    data_test.class_is_last()
    N_test = data_test.num_instances

    res,my_score = test(data_test,N_test,attributes_test,clfs,w1,b1,w2_init)

    res2,my_score2 = test(data_test,N_test,attributes_test,clfs,w1_init,b1_init,w2_init)

    print('Score using initial values: ',my_score2)

    jvm.stop()
