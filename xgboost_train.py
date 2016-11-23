import numpy as np
## version 2 0.983843

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
import evaluation

from sklearn.preprocessing import StandardScaler


offset = 10000

print("Load the training/test data using pandas")
train = pd.read_csv("training.csv")
test  = pd.read_csv("test.csv")
check_agreement = pd.read_csv('check_agreement.csv')
check_correlation = pd.read_csv('check_correlation.csv')

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

train = np.array(train)
np.random.seed(671)
np.random.shuffle(train)


print("Train a UGradientBoostingClassifier")

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.02,# used to be 0.2 or 0.1
          "max_depth": 9, # used to be 5 or 6
          "min_child_weight": 1,
          "silent": 1,
          "colsample_bytree": 0.7,
		  "subsample": 0.8,
		  "eval_metric" : "logloss",
          "seed": 1}
num_trees=2000 #used to be 300, 375 is better
print train[:,0:46]
print train[:,48]

labels = train[:,48]
train = train[:,1:46]

xgtrain = xgb.DMatrix(train[offset:,:],labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], labels[:offset])
watchlist = [(xgtrain, 'train'),(xgval, 'val')]

xgtest = xgb.DMatrix(test[features])

model = xgb.train(params, xgtrain, num_trees, watchlist, early_stopping_rounds=100)
#gbm = xgb.train(params, xgtrain, num_trees, watchlist)
preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)

pred_aggreement = model.predict(xgb.DMatrix(check_agreement[features]),ntree_limit=model.best_iteration)
pred_correlation = model.predict(xgb.DMatrix(check_correlation[features]),ntree_limit=model.best_iteration)

ks = evaluation.compute_ks(pred_aggreement[check_agreement['signal'].values == 0], pred_aggreement[check_agreement['signal'].values == 1], check_agreement[check_agreement['signal'] == 0]['weight'].values, check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)

print ('Checking correlation...')
cvm = evaluation.compute_cvm(pred_correlation, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)


print("Make predictions on the test set")
submission = pd.DataFrame({"id": test["id"], "prediction": preds1})
submission.to_csv("xgboost_sol_preds1.csv", index=False)

r = random.sample(xrange(0,train.shape[0]),offset)
#create a train and validation dmatrices
train_val = train[r,:]
labels_val = labels[r]
print "\nsize of train_val is",train_val.shape[0]," ",train_val.shape[1]
train_new = np.delete(train, r, axis = 0)
labels_new = np.delete(labels, r)
#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train_new, label=labels_new)
xgval = xgb.DMatrix(train_val, label=labels_val)
#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(params, xgtrain, num_trees, watchlist, early_stopping_rounds=100)
preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)

print("Make predictions on the test set")
submission = pd.DataFrame({"id": test["id"], "prediction": preds2})
submission.to_csv("xgboost_sol_preds2.csv", index=False)

r = random.sample(xrange(0,train.shape[0]),offset)
#create a train and validation dmatrices
train_val = train[r,:]
labels_val = labels[r]
print "\nsize of train_val is",train_val.shape[0]," ",train_val.shape[1]
train_new = np.delete(train, r, axis = 0)
labels_new = np.delete(labels, r)
#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train_new, label=labels_new)
xgval = xgb.DMatrix(train_val, label=labels_val)
#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(params, xgtrain, num_trees, watchlist, early_stopping_rounds=100)
preds3 = model.predict(xgtest,ntree_limit=model.best_iteration)
print("Make predictions on the test set")
submission = pd.DataFrame({"id": test["id"], "prediction": preds3})
submission.to_csv("xgboost_sol_preds3.csv", index=False)


r = random.sample(xrange(0,train.shape[0]),offset)
#create a train and validation dmatrices
train_val = train[r,:]
labels_val = labels[r]
print "\nsize of train_val is",train_val.shape[0]," ",train_val.shape[1]
train_new = np.delete(train, r, axis = 0)
labels_new = np.delete(labels, r)
#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train_new, label=labels_new)
xgval = xgb.DMatrix(train_val, label=labels_val)
#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(params, xgtrain, num_trees, watchlist, early_stopping_rounds=100)
preds4 = model.predict(xgtest,ntree_limit=model.best_iteration)
print("Make predictions on the test set")
submission = pd.DataFrame({"id": test["id"], "prediction": preds4})
submission.to_csv("xgboost_sol_preds4.csv", index=False)

preds = (preds1+preds2+preds3+preds4)/4

print("Make predictions on the test set")
submission = pd.DataFrame({"id": test["id"], "prediction": preds})
submission.to_csv("xgboost_sol.csv", index=False)
