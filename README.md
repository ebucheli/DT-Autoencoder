# DT-Autoencoder

### Usage

Four possible input arguments:

 1. `--training_data`: arff file with training data, check Input Data section for more info.
 1. `--testing_data`: arff file with testing data, check Input Data section for more info.
 1. `--outfile`: json file that stores results with trained and initial weights.
 1. `--rand_init`: use this flag to use random weight initialization.

Use `--rand_init` flag to start with random parameters, Recall and AUC otherwise.

##### Example:

The following command will train the model using the `breast-cancer.training1.arff` file and test using `breast-cancer.testing1.arff`. The results will be saved in `my_result.json`. The weights will be randomly initialized.

`python main.py --training_data breast-cancer.training1.arff --testing_data breast-cancer.testing1.arff --outfile my_result.json --rand_init`

### Dependencies

 1. python=3.6.7
 1. [python-weka-wrapper3](https://pypi.org/project/python-weka-wrapper3/)
 1. Anaconda

### Input Data

Only for categorical variables.

Training dataset should contain Class Attribute last, should be only one class (e.g. `genuine`).

Testing dataset should contain Class attribute last, should be binary labels (e.g. `impostor`, `genuine` but any name will work as long as it is binary). `impostor` Label should be first.
