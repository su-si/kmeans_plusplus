
import numpy as np
#import random as rd

class Dataset():
    def __init__(self, X, y, labels=None):
        ''' :param labels: a dict matching integers in y to class names'''
        self.X = X
        self.y = y
        self.labels = labels


def iris_labels():
    return {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


def load_iris_data(seed=None, val_percent = 0.2):
    ''' load labels and data, split it into train and validation set
        :param seed: if not None, numpy.random will be seeded to this value.
                    Influences train-val split choice.
    '''
    #datafile = "/home/noobuntu/Repos/programming_task_mpg/data/iris/iris.data"
    datafile = "./data/iris/iris.data"
    X = np.zeros((150,4)) # X_list
    y = np.zeros(150, dtype=np.int8)
    label_dict = iris_labels()
    with open(datafile, 'r') as fd:
        lines = fd.readlines()
        for i, line in enumerate(lines):
                   # 5.1,3.5,1.4,0.2,Iris-setosa\n
            vals = line.strip("\n") .split(",")
            xvals = [float(x) for x in vals[:-1]]
            if len(xvals) > 0:
                X[i,:] = xvals
                y[i] = label_dict[vals[-1]]

    if seed is not None:
        np.random.seed(seed)
    p = np.random.permutation(len(y))
    val_idx_start = int((1.-val_percent) * len(y))
    ds_train =  Dataset(X=X[p[:val_idx_start]], y=y[p[:val_idx_start]], labels=iris_labels())
    ds_val =  Dataset(X=X[p[val_idx_start:]], y=y[p[val_idx_start:]], labels=iris_labels())

    return ds_train, ds_val





if __name__ == "__main__":

    ds_train, ds_val = load_iris_data()
    print("number of train samples: "+str(len(ds_train.X)))
    print("number of val samples: "+str(len(ds_val.X)))

