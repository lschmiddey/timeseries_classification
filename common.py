import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools


def save_obj(obj: str, name: str):
    """Save object as a pickle file to a given path."""
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name: str):
    """Load object as a pickle file to a given path."""
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def normalize(x, m, s): return (x-m)/s


def cat_transform(train_var: np.array, test_var: np.array):
    """remap number to categorical variable and save dictionaries.
    test_var is then mapped according to the train_var"""
    dict_list = []
    train_var_shape = train_var.shape
    test_var_shape = test_var.shape
    if len(train_var.shape)==1:
        train_var = train_var.reshape(-1,1)
        test_var = test_var.reshape(-1,1)
    for i in range(train_var.shape[1]):
        dict_ = {j: element for j, element in enumerate(set(train_var[:,i]))}
        dict_list.append(dict_)
    dict_inv_list = [{v: k for k, v in dict_list[i].items()}
                     for i, dict_ in enumerate(dict_list)]

    # map numpy arrays
    for i in range(train_var.shape[1]):
        train_var[:,i] = np.vectorize(dict_inv_list[i].get)(train_var[:,i])
    for i in range(test_var.shape[1]):
        test_var[:,i] = np.vectorize(dict_inv_list[i].get)(test_var[:,i])

    train_var = train_var.reshape(train_var_shape).astype(int)
    test_var = test_var.reshape(test_var_shape).astype(int)

    return train_var, test_var, dict_list, dict_inv_list


def cat_transform_reverse(var: np.array, dict_list:list):
    """reverses the number to the correct category"""
    var_shape = var.shape
    var = var.reshape(-1,1)
    for i in range(var.shape[1]):
        var[:,i] = np.vectorize(dict_list[i].get)(var[:,i])
    var = var.reshape(var_shape)
    return var


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()