__author__ = 'rhegde'

import numpy as np
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import  precision_score
from sklearn.metrics import  recall_score
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import  roc_auc_score


# load the CSV file as a numpy matrix


dataset = np.loadtxt(sys.argv[3],delimiter=",")

features = dataset[:, 1:12]
#60 -40 dataset split (Training data 60 %, Testing data 40 %)
training_features, test_features = features[:2939], features[2939:]

print("Training Features", training_features.shape,training_features)
print("Test Features", test_features.shape,test_features)


target =  np.array(dataset[:, :1]).ravel()

training_target, test_target = target[:2939], target[2939:]

print("Training target", training_target.shape, training_target)
print("Test target", test_target.shape,test_target)


clf = SGDClassifier(loss=sys.argv[1], penalty=sys.argv[2], shuffle=False, n_iter=200, fit_intercept=True, alpha=.01)



clf.fit(training_features,training_target)
training_predictions = clf.predict(training_features)

print("######### Training Data Metrics #####################")


print("Training Accuracy Score: ",accuracy_score(training_target,training_predictions))
print("Training Precision Score: ",precision_score(training_target,training_predictions))
print("Training Recall Score: ",recall_score(training_target,training_predictions))
print("Training AUC ROC Score : ",roc_auc_score(training_target,training_predictions))
print("Training Confusion Matrix: ",confusion_matrix(training_target,training_predictions))




print("######### Test Data Metrics #####################")
test_predictions = clf.predict(test_features)

print("Test Accuracy Score: ",accuracy_score(test_target,test_predictions))
print("Test Precision Score: ",precision_score(test_target,test_predictions))
print("Test Recall Score: ",recall_score(test_target,test_predictions))
print("Test AUC ROC Score : ",roc_auc_score(test_target,test_predictions))
print("Test Confusion Matrix: ",confusion_matrix(test_target,test_predictions))



