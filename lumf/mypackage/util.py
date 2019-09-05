import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

def compute_f1_maxscore(y_test, pred_test_y):
    """
    return max f1_score
    """
    opt_prob = None
    f1_max = 0
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))
        print('F1 score at threshold {} is {}'.format(thresh, f1))
        
        if f1 > f1_max:
            f1_max = f1
            opt_prob = thresh
    print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
    return f1_max

def compute_auc(label_y, pred_y_prob):
    """
    type = {0, 1}
    """
    fprs, tprs, thresholds = metrics.roc_curve(label_y, pred_y_prob)
    
    for i,  (fpr, tpr, thres) in enumerate(zip(fprs, tprs, thresholds)):
        print("The {} thresh value={} computes fpr={}, tpr={}".format(i, thres, fpr, tpr))
    auc = metrics.auc(fprs, tprs) 
    print("auc = ", auc)
    return auc


def convert_prob_to_label(label_probs, prob_threshold:float,  positive_label:int,  negative_label:int):
    labels = [ positive_label if prob > prob_threshold  else  negative_label for prob in label_probs ]
    return labels

def onehot_postpad_docs(docs : pd.Series, vocab_size : int, doc_words_maxlen: int):
    """
    make onehot and pad change on docs, and be setted as input for embed layer
    """
    onehot_docs = [one_hot(doc, vocab_size) for doc in docs ]
    padded_docs = pad_sequences(onehot_docs, maxlen = doc_words_maxlen)
    print('onehot and padded shape : {} '.format(padded_docs.shape))
    return padded_docs
    