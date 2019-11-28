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

from classifier import train_trees, get_initial_weights,sigmoid,neuron_l1,train
from data_loader import load_data, make_partition

if __name__ == '__main__':

    jvm.start()

    data, attributes = load_data('./breast-cancer.arff')
    data_normal, N = make_partition(data,attributes)

    clfs,dt_y_hat = train_trees(data_normal,attributes)

    #for clf in clfs:
        #print(clf)

    w2_init,w1_init,b1_init = get_initial_weights(data_normal,clfs,attributes,dt_y_hat)

    #print(w2_init,w1_init,bias_init)

    w1,b1=train(w1_init = w1_init,
                b1_init = b1_init,
                lr = 0.01,
                iterations = 300,
                N = N,
                data = data_normal,
                attributes = attributes,
                dt_y_hat = dt_y_hat)

    jvm.stop()
