import numpy as np
from sklearn.svm import SVC as svm_cl

def fit_and_classify(train_features, train_labels, test_features):
    model = svm_cl(C=160)
    model.fit(train_features, train_labels)
    prdct = model.predict(test_features)
    return prdct
