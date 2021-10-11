
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def sensitivity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)

def specificity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)

def auc(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob)

def mcc(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)

def accuracy(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)

def precision(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.precision_score(y_true,y_prob)

def recall(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.recall_score(y_true,y_prob)

def f1(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.f1_score(y_true,y_prob)

def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)

def cofusion_matrix(y_true,y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()

    return tn, fp, fn, tp
































