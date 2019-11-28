from weka.core import jvm
from weka.core.converters import Loader
from weka.filters import Filter

import numpy as np
import pandas as pd

def load_data(filename):
    loader = Loader(classname = 'weka.core.converters.ArffLoader')
    data = loader.load_file(filename)
    attributes = [f.name for f in data.attributes()]

    return data, attributes

def make_partition(data,attributes,part = 'normal'):

    if part == 'normal':
        value = 'last'
    elif part == 'anomalous':
        value = 'first'

    keep_normal = Filter(classname='weka.filters.unsupervised.instance.RemoveWithValues',
                         options = ['-C','last','-L',value])
    keep_normal.inputformat(data)
    data_normal = keep_normal.filter(data)

    remove = Filter(classname='weka.filters.unsupervised.attribute.Remove',
                    options = ['-R','last'])
    remove.inputformat(data)
    data_normal = remove.filter(data_normal)

    N = data_normal.num_instances

    return data_normal, N
