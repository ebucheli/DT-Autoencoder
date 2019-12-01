import argparse
import json

import numpy as np
#import pandas as pd
from scipy.io import arff

from weka.core import jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.filters import Filter

from copy import deepcopy

from classifier import train_trees, get_initial_weights,sigmoid,neuron_l1,train, test
from data_loader import load_data, make_partition

parser = argparse.ArgumentParser()
parser.add_argument('--training_data', dest = 'train_path')
parser.add_argument('--testing_data', dest = 'test_path')
parser.add_argument('--rand_init', action= 'store_true')
parser.add_argument('--outfile', dest = 'outfile')

args = parser.parse_args()

if __name__ == '__main__':

    train_path = args.train_path
    test_path = args.test_path
    rand_init = args.rand_init
    outfile = args.outfile

    try:

        jvm.start()

        data_train, attributes = load_data(train_path)
        N_train = data_train.num_instances
        attributes = attributes[:-1]

        remove = Filter(classname='weka.filters.unsupervised.attribute.Remove',
                        options = ['-R','last'])
        remove.inputformat(data_train)
        data_train = remove.filter(data_train)

        doit = False
        to_del = []

        for i in range(data_train.num_attributes):
            if len(data_train.attribute(i).values) <= 1:
                to_del.append(i+1)
                doit = True

        if doit:
            to_del_str = ','.join(str(e) for e in to_del)
            remove_const = Filter(classname='weka.filters.unsupervised.attribute.Remove',
                                  options = ['-R',to_del_str])
            remove_const.inputformat(data_train)
            data_train = remove_const.filter(data_train)

        attributes = [f.name for f in data_train.attributes()]
        m_train = data_train.num_attributes

        print('\n\n############ Training Decision Trees ############\n\n')

        clfs,evls,dt_y_hat,unused_attributes = train_trees(data_train,attributes)

        w2_init,w1_init,b1_init = get_initial_weights(data_train,
                                                      clfs,
                                                      evls,
                                                      attributes,
                                                      dt_y_hat,
                                                      unused_attributes)

        if rand_init == True:

            w1_init = []
            for i in range(m_train):
                len_values = len(data_train.attribute(i).values)
                w1_init.append(np.random.randn(len_values))

        print('Done!')

        print('\n\n############ Training Hidden Layer Parameters ############\n\n')

        if rand_init:
            print('Weights Randomly Initialized\n\n')
        else:
            print('Weights initialized with Recall\n\n')


        w1,b1 = train(w1_init = w1_init,
                      b1_init = b1_init,
                      lr = 0.001,
                      epochs = 5,
                      N = N_train,
                      data = data_train,
                      attributes = attributes,
                      dt_y_hat = dt_y_hat,
                      batch_size = 32,
                      unused_attributes = unused_attributes)

        data_test,attributes_test = load_data(test_path)
        data_test.class_is_last()
        N_test = data_test.num_instances

        doit = False
        to_del = []

        for i in range(data_test.num_attributes):
            if len(data_test.attribute(i).values) <= 1:
                to_del.append(i+1)
                doit = True

        if doit:
            print('Deleting Constant Attributes {}'.format(to_del))
            to_del_str = ','.join(str(e) for e in to_del)
            remove_const = Filter(classname='weka.filters.unsupervised.attribute.Remove',
                                  options = ['-R',to_del_str])
            remove_const.inputformat(data_test)
            data_test = remove_const.filter(data_test)

        attributes_test = [f.name for f in data_test.attributes()]

        print('\n\n############ Running Model on Test Data ############\n\n')

        res,my_score = test(data_test,
                            N_test,
                            attributes_test,
                            clfs,
                            w1,
                            b1,
                            w2_init,
                            unused_attributes)


        print('\n\nAUC using trained values: {}'.format(my_score))

        res2,my_score2 = test(data_test,
                              N_test,
                              attributes_test,
                              clfs,
                              w1_init,
                              b1_init,
                              w2_init,
                              unused_attributes,
                              verbose = 0)

        print('\n\nAUC using initial values: {}\n\n'.format(my_score2))

        dict_res = {'AUC_trained':my_score,
                    'scores_trained':res.tolist(),
                    'AUC_initial':my_score2,
                    'scores_initial':res2.tolist()}

        with open(outfile, 'w') as fp:
            json.dump(dict_res, fp)

    finally:
        jvm.stop()
