import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.io import arff

from sklearn.metrics import recall_score, roc_auc_score

from weka.core import jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.filters import Filter

from copy import deepcopy
#import sklearn


def train_trees(data,attributes):

    clfs = []
    dt_y_hat = []

    for i,att in enumerate(attributes[:-1]):

        data.class_index = i

        this_clf = Classifier(classname='weka.classifiers.trees.J48',options = ['-C','0.2','-M','2'])

        this_clf.build_classifier(data)

        dt_y_hat.append(this_clf.distributions_for_instances(data))

        clfs.append(this_clf)

    return clfs,dt_y_hat

def get_initial_weights(data,clfs,attributes,dt_y_hat):
    w1_init = []
    w2_init = []

    for i,att in enumerate(attributes[:-1]):

        this_y_hat = np.argmax(dt_y_hat[i],axis = 1)
        this_y = data.values(i)

        rocs = []

        for j in np.unique(this_y):

            new_y_hat = np.array([1 if f == j else 0 for f in this_y_hat])
            new_y = np.array([1 if f == j else 0 for f in this_y])

            if np.all(new_y == 0):
                pass
            else:
                rocs.append(roc_auc_score(new_y,new_y_hat))

        w2_init.append(np.mean(rocs))

        print('AUC: {:0.4f}'.format(np.mean(rocs)))

        temp_w = np.zeros(len(data.attribute(i).values))

        this_y[np.isnan(this_y)] = 0

        this_recs = recall_score(this_y,this_y_hat,average=None)

        for i,rec in zip(np.unique(np.concatenate((this_y_hat,this_y))),this_recs):

            temp_w[int(i)] = rec

        w1_init.append(temp_w)

    bias_init = np.zeros((len(w2_init)))

    return w2_init,w1_init,bias_init

def sigmoid(x):
    return 1/(1+np.exp(-x))

def neuron_l1(x_prime,weights,bias,indxs):

    my_res = np.zeros((len(x_prime)))

    for i,this_x_prime in enumerate(x_prime):

        if indxs[i] >= len(weights):
            indxs[i] -= 1
        this_x_wrong = np.delete(this_x_prime,indxs[i])
        w_wrong = np.delete(weights,indxs[i])

        my_res[i] = sigmoid(this_x_prime[indxs[i]]*weights[indxs[i]]-np.mean(this_x_wrong*w_wrong)+bias)

    return my_res

def train(w1_init,b1_init,lr,iterations,N,data,attributes,dt_y_hat):

    w1 = deepcopy(w1_init)
    b1 = deepcopy(b1_init)

    losses = []
    accs = []

    for i in range(iterations):

        hl1_this = np.zeros((N,len(dt_y_hat)))

        for j,x_prime in enumerate(dt_y_hat):

            x_prime_prime = x_prime*w1[j]
            num_labels = np.array(data.values(j),dtype = np.int64)

            for dani,f in enumerate(num_labels):
                if f < 0:
                    num_labels[dani] = 0

            for k in np.unique(num_labels):

                if np.isnan(k):
                    k = 0


                indices = np.where(num_labels==k)
                this_probs = x_prime[indices]
                x_prime_this = x_prime[indices]
                x_prime_prime_this = x_prime_prime[indices]

                a = neuron_l1(x_prime_this,w1[j],b1[j],np.ones((len(x_prime_this)),dtype = int)*k)

                grad_wj = np.dot(a*(1-a),x_prime_this[:,k])
                grad_bias = np.mean(a*(1-a))


                w1[j][k] = w1[j][k] + lr*grad_wj
                b1[j] = b1[j] + lr*grad_bias
                #for this_indx in [f for f in np.arange(np.max(num_labels)) if f != k]:

                #    if this_indx < k:
                #        grad_indx = this_indx
                #    elif this_indx > k:
                #        grad_indx = this_indx-1

                #    w1[j][this_indx] = w1[j][grad_indx] - lrw1*grad_wl[grad_indx]


        for j,x_prime in enumerate(dt_y_hat):

            num_labels = np.array(data.values(j),dtype = np.int64)

            for dani,f in enumerate(num_labels):
                if f < 0:
                    num_labels[dani] = 0

            hl1_this[:,j] = neuron_l1(x_prime,w1[j],b1[j],num_labels)

        this_loss = np.mean(hl1_this,axis = 0)
        if i % 50 == 0 or i == iterations-1:
            print('Iteration {}:'.format(i+1))
            for m, loss_part in enumerate(this_loss):
                print('\tAttribute {}: {:0.4f}'.format(m+1,loss_part))
    return w1,b1
