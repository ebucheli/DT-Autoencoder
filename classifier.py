import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#from scipy.io import arff

from sklearn.metrics import recall_score, roc_auc_score

from weka.core import jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter

from copy import deepcopy
#import sklearn


def train_trees(data,attributes):

    clfs = []
    evls = []
    dt_y_hat = []

    for i,att in enumerate(attributes):

        data.class_index = i

        this_clf = Classifier(classname='weka.classifiers.trees.J48',options = ['-C','0.25','-M','2'])
        this_clf.build_classifier(data)

        this_evl = Evaluation(data)
        this_evl.crossvalidate_model(this_clf,data,5,Random(1))

        dt_y_hat.append(this_clf.distributions_for_instances(data))
        clfs.append(this_clf)
        evls.append(this_evl)

    return clfs,evls,dt_y_hat

def get_initial_weights(data,clfs,evls,attributes,dt_y_hat):
    w1_init = []
    w2_init = []

    for i,att in enumerate(attributes):

        print('Attribute: {}\n'.format(att))

        rocs = []

        att_values = data.attribute(i).values
        len_att_values = len(att_values)

        temp_w = np.zeros(len_att_values)

        for j in range(len_att_values):

            this_roc = evls[i].area_under_roc(j)

            if np.isnan(this_roc):
                print('\tNAN at value {}'.format(data.attribute(i).values[j]))
                this_roc = 1

            rocs.append(this_roc)

            temp = evls[i].recall(j)
            if np.isnan(temp):
                temp = 0
            temp_w[j] = temp

        print('\tAverage AUC: {:0.4f}\n'.format(np.mean(rocs)))

        w1_init.append(temp_w)
        w2_init.append(np.mean(rocs))

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
                print('\tAttribute {} Loss: {:0.4f}'.format(m+1,loss_part))
    return w1,b1

def get_batches(dt_y_hat,batch_size=32):

    N = len(dt_y_hat)

    if N < batch_size:
        batch_size = N

    reps = N // batch_size

    batches = []
    batch_startend = []

    for i in range(reps):
        start = i*batch_size
        end = start+batch_size
        batches.append(dt_y_hat[start:end])
        batch_startend.append([start,end])
    if end < N:
        start = end
        end = N
        batches.append(dt_y_hat[start:end])
        batch_startend.append([start,end])

    return batches,batch_startend

def train_v2(w1_init,b1_init,lr,epochs,N,data,attributes,dt_y_hat,batch_size):

    w1 = deepcopy(w1_init)
    b1 = deepcopy(b1_init)

    losses = []
    accs = []

    for epoch in range(epochs):

        hl1_this = np.zeros((N,len(dt_y_hat)))

        for j,x_prime in enumerate(dt_y_hat):

            batches,batch_startend = get_batches(x_prime,batch_size=batch_size)
            num_labels = np.array(data.values(j),dtype = np.int64)

            for batch_ind, x_prime_batch in enumerate(batches):
                x_prime_prime_batch = x_prime_batch*w1[j]
                start,end = batch_startend[batch_ind]
                batch_num_labels = num_labels[start:end]

                for dani,f in enumerate(batch_num_labels):
                    if f < 0:
                        batch_num_labels[dani] = 0

                for k in np.unique(batch_num_labels):

                    if np.isnan(k):
                        k = 0

                    indices = np.where(batch_num_labels==k)
                    this_probs = x_prime_batch[indices]
                    x_prime_this = x_prime_batch[indices]
                    x_prime_prime_this = x_prime_prime_batch[indices]

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
        if epoch % 10 == 0 or epoch == epochs-1:
            print('Epoch {}:'.format(epoch+1))
            for m, loss_part in enumerate(this_loss):
                print('\tAttribute {} Loss: {:0.4f}'.format(m+1,loss_part))
    return w1,b1

def test(data,N,attributes,clfs,w1,b1,w2):

    remove = Filter(classname='weka.filters.unsupervised.attribute.Remove',
                    options = ['-R','last'])
    remove.inputformat(data)

    data_noclass = remove.filter(data)

    dt_all = []

    for i,att in enumerate(attributes[:-1]):
        data_noclass.class_index = i
        dt_all.append(clfs[i].distributions_for_instances(data_noclass))

    hl1_this_all = np.zeros((N,len(attributes[:-1])))
    preds = []

    for j,x_prime in enumerate(dt_all):

        num_labels = np.array(data_noclass.values(j),dtype = np.int64)

        for dani,f in enumerate(num_labels):
            if f < 0:
                num_labels[dani] = 0

        preds.append(neuron_l1(x_prime,w1[j],b1[j],num_labels))

    res = np.dot(np.array(preds).T,w2)/len(attributes[:-1])

    class_index_data = data.class_index

    y = data.values(class_index_data)
    y = np.abs(y-1)

    #print(y)
    #print(res)
    my_score = roc_auc_score(y,res)

    return res,my_score
