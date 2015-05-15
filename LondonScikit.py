print(__doc__)
'''london-scikit competions'''
'''@author Elena Cuoco'''
import numpy as np
import pandas as pd
import os
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

if __name__ == '__main__':
 os.chdir('./kaggle/scikit/2014')

 train_data = pd.read_csv('../data/train.csv', header=None)
 train_labels = pd.read_csv('../data/trainLabels.csv', header=None)
 test_data = pd.read_csv('../data/test.csv', header=None)
 X_train = np.asarray(train_data)
 Y_train = np.asarray(train_labels).ravel()
 X_test= np.asarray(test_data)


 X_all=np.r_[X_train, X_test]
 lowest_bic = np.infty
 bic = []
 n_components_range = range(1, 7)

 cv_types = ['spherical', 'tied', 'diag', 'full']
 for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GMM(n_components=n_components, covariance_type=cv_type,min_covar=0.0001,thresh=0.1)

        gmm.fit(X_all)
        bic.append(gmm.aic(X_all))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
 g=best_gmm
 g.fit(X_all)

 X =g.predict_proba(X_train)

 clf=RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=1,
  min_samples_leaf=3, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=33,
  verbose=0, min_density=None, compute_importances=None)


 ###compute grid search to find best paramters for pipeline
 param_grid = dict( )

 grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=3,scoring='accuracy',cv=5).fit(X, Y_train)
 ###print result
 print(grid_search.best_estimator_)
 report(grid_search.grid_scores_)


 ###predict best fit
 svc=grid_search.best_estimator_.fit(X,Y_train)

 print ('best'), grid_search.best_estimator_.score(X, Y_train)
 scores = cross_val_score(svc, X, Y_train,cv=5,scoring='accuracy')
 print scores.mean(),scores.min()
 print scores

 X_t =g.predict_proba(X_test)

 y_pred=grid_search.best_estimator_.predict(X_t)
 print y_pred.shape
 print g

 data=[]
 for i in range(9000):
  data.append([i+1,y_pred[i]])

 with open('predict-pgq-better-2014-cleaned.csv', 'wb') as csvfile:
    out = csv.writer(csvfile, delimiter=',')
    out.writerow(['Id','Solution'])
    out.writerows(data)
