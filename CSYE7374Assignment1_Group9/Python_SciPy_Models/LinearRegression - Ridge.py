__author__ = 'rhegde'

import numpy as np
import sys
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import  mean_squared_error




# load the CSV file as a numpy matrix


dataset = np.loadtxt(sys.argv[1],delimiter=",")

features = dataset[:, 1:12]
#60 -40 dataset split (Training data 60 %, Testing data 40 %)
training_features, test_features = features[:2939], features[2939:]

print("Training Features", training_features.shape,training_features)
print("Test Features", test_features.shape,test_features)


target =  np.array(dataset[:, :1]).ravel()

training_target, test_target = target[:2939], target[2939:]

print("Training target", training_target.shape, training_target)
print("Test target", test_target.shape,test_target)


#clf = LinearRegression(fit_intercept=True)

#clf = SGDRegressor( penalty='none', alpha=0.0001, fit_intercept=True, n_iter=200, shuffle=False)
clf = Ridge(alpha=.0001, fit_intercept=True, max_iter=200, tol=0.001)
#clf = Lasso(alpha=0.001, fit_intercept=True, max_iter=200, tol=0.001)

clf.fit(training_features,training_target)
training_predictions = clf.predict(training_features)

print("######### Training Data Metrics #####################")


print("Training RMS Error: ",mean_squared_error(training_target,training_predictions))




print("######### Test Data Metrics #####################")
test_predictions = clf.predict(test_features)

print("Test RMS Error: ",mean_squared_error(test_target,test_predictions))


