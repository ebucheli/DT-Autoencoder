# DT-Autoencoder

### Usage

Four possible input arguments:

 1. `--training_data`: arff file with training data, check Input Data section for more info.
 1. `--testing_data`: arff file with testing data, check Input Data section for more info.
 1. `--outfile`: json file that stores results with trained and initial weights.
 1. `--rand_init`: use this flag to use random weight initialization.

Use `--rand_init` flag to start with random parameters, Recall and AUC otherwise.

`python main.py --training_data [train_data].arff --testing_data [test_data].arff --outfile [outfile].json --rand_init`

### Dependencies

 1. python=3.6.7
 1. [python-weka-wrapper3](https://pypi.org/project/python-weka-wrapper3/)
 1. Anaconda

### Input Data

Only for categorical variables.

Training dataset should not contain original Class attribute. Should contain only one class. 

Testing dataset should contain Class attribute last, should be binary labels (eg. `normal`, `anomaly` but any name will work as long as it is binary). `Normal` Label should be first.
