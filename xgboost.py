#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from xgboost import plot_importance
# from sklearn import model_selection, metrics  # Additional scklearn functions
# from sklearn.grid_search import GridSearchCV  # Perforing grid search
# from sklearn import cross_validation
# from sklearn.cross_validation import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def feat_imp(df, model, n_features):
	d = dict(zip(df.columns, model.feature_importances_))
	ss = sorted(d, key=d.get, reverse=True)
	top_names = ss[0:n_features]

	plt.figure(figsize=(15, 15))
	plt.title("Feature importances")
	plt.bar(range(n_features), [d[i] for i in top_names], color="b", align="center")
	plt.xlim(-1, n_features)
	plt.xticks(range(n_features), top_names, rotation='vertical')
	plt.gcf().savefig('feature_importance_xgb.png')
	plt.show()


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
						  metrics='auc', early_stopping_rounds=early_stopping_rounds)
		alg.set_params(n_estimators=cvresult.shape[0])

	# Fit the algorithm on the data
	booster_ = alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

	# Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

	# Predict testing set:
	# dtest_predictions = alg.predict(dtest[predictors])
	dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
	precision, recall, _ = precision_recall_curve(dtest[target], dtest_predprob)
	average_precision = average_precision_score(dtest[target], dtest_predprob)
	# Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)
	print "AUC Score (Test): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob)

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
	plt.gcf().savefig('Precision-Recall curve.png')
	plt.show()
	feat_imp(dtrain, booster_, 10)


# plot_importance(booster_,max_num_features=10)
# plt.gcf().savefig('feature_importance_xgb.png')
# plt.show()


if __name__ == '__main__':

	train = pd.read_csv('c:/train_modified.csv')
	dtrain, dtest = train_test_split(train, test_size=0.3, random_state=0)
	target = 'Disbursed'
	IDcol = 'ID'
	features = [x for x in train.columns if x not in [target, IDcol]]
	xgbc = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
						 colsample_bytree=0.8, reg_alpha=0.01, objective='binary:logistic', nthread=4,
						 scale_pos_weight=1, seed=27)
	modelfit(xgbc, dtrain, dtest, features)
