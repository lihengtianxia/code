import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from xgboost.sklearn import XGBClassifier
# from sklearn import model_selection, metrics  # Additional scklearn functions
# from sklearn.grid_search import GridSearchCV  # Perforing grid search
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
# from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

train = pd.read_csv('./train_modified.csv')
dtrain,dtest=train_test_split(train,test_size=0.3,random_state=0)

target = 'Disbursed'
IDcol = 'ID'
xgb_params = {"booster":"gbtree","objective":"binary:logistic", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}

# Choose all predictors except target & IDcols
features = [x for x in train.columns if x not in [target, IDcol]]
ceate_feature_map(features)

dtrains = xgb.DMatrix(dtrain[features].values, label=dtrain[target].values)
dtests = xgb.DMatrix(dtest[features].values, label=dtest[target].values)
watchlist = [(dtests, 'eval'), (dtrains, 'train')]
num_rounds = 2


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)



xgbc = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgbc, train, features)


bst = xgb.train(xgb_params, dtrains, num_rounds, watchlist)
# feat_impt = pd.Series(bst.get_fscore()).sort_values(ascending=False)
# feat_impt.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.show()

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
plt.show()
