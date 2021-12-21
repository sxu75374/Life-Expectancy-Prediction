'''
Project Coding Part - Life Expectancy Prediction by Regression - main.py file
Entirely done by
sxu75374
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
import pycountry_convert as pc
from scipy.stats import pearsonr
import seaborn as sns
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings(action='ignore')
plt.ion()
import sys
sys.path.append("/Users/xs/PycharmProjects/EE660FinalProject/TrAdaBoost1.py")


def trivial_system(X, y):
    '''calculate the mean value of Life Expectancy as the trivial system to compare'''
    ypred = [np.mean(y)]*len(y)
    return ypred


def non_trivial_system(Xtrain, ytrain, Xtest, ytest):
    '''use the basic LinearRegression() model as the baseline model'''
    baseline = LinearRegression()
    baseline.fit(Xtrain, ytrain)
    ypred = baseline.predict(Xtest)
    return ypred


def label_propagation_regression(X_l, y_l, X_u, X_val, y_val, nn, pi, sigma_2):
    '''
    Thanks for user ermongroup for providing code of Label Propagation
    https://github.com/ermongroup/ssdkl/tree/master/labelprop_and_meanteacher
    '''
    # concatenate all the X's and y's
    X_all = np.concatenate([X_l, X_u, X_val], axis=0)
    knn = KNeighborsRegressor(n_neighbors=nn, p=pi,weights='distance')
    knn.fit(X_l, y_l)
    y_u = knn.predict(X_u)
    y_val_pred = knn.predict(X_val)
    y_all = np.concatenate([y_l, y_u, y_val_pred])

    # compute the kernel
    T = np.exp(-cdist(X_all, X_all, metric='sqeuclidean') / sigma_2)
    # row normalize the kernel
    T /= np.sum(T, axis=1)[:, np.newaxis]

    delta = np.inf
    tol = 3e-6
    i = 0
    while delta > tol:
        y_all_new = T.dot(y_all)
        # clamp the labels known
        y_all_new[:X_l.shape[0]] = y_l
        delta = np.mean(y_all_new - y_all)
        y_all = y_all_new
        i += 1
        val_loss = np.mean(np.square(y_all[-X_val.shape[0]:] - y_val))
        if i % 10 == 0:
            print("Iter {}: delta={}, val_loss={}".format(i, delta, val_loss))
        if i > 500:
            break
    # return final val loss (MSE)
    return val_loss, y_val_pred


print('\n')
print('================================ Supervised Learning Part ================================')
import pickle
# load SL test set
# save np.load
X_test_SL = np.load('data/X_test_SL.npy')

# modify the default parameters of np.load
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
Y_test_SL = np.load('data/Y_test_SL.npy')
np.load = np_load_old

# load models
filename0 = 'models/non_trivial_system.sav'
non_trivial_system = pickle.load(open(filename0, 'rb'))
filename1 = 'models/best_lasso.sav'
best_lasso = pickle.load(open(filename1, 'rb'))
filename2 = 'models/best_ridge.sav'
best_ridge = pickle.load(open(filename2, 'rb'))
filename3 = 'models/best_cart.sav'
best_cart = pickle.load(open(filename3, 'rb'))
filename4 = 'models/best_rf.sav'
best_rf = pickle.load(open(filename4, 'rb'))
filename5 = 'models/best_ada.sav'
best_ada = pickle.load(open(filename5, 'rb'))
filename6 = 'models/best_knn.sav'
best_knn = pickle.load(open(filename6, 'rb'))
filename7 = 'models/best_svr.sav'
best_svr = pickle.load(open(filename7, 'rb'))


# model 0: trivial system
print('=== Model 0: trivial system ===')
pred_trivial = trivial_system(X_test_SL, Y_test_SL)
mse_trivial = mean_squared_error(Y_test_SL, pred_trivial)
print('Test MSE of Baseline trivial system: ', mse_trivial)
print('Test R^2 of Baseline trivial system: ', r2_score(Y_test_SL, pred_trivial))

# model 0: non-trivial system
print('\n')
print('=== Model 0: non-trivial system (Linear Regression) ===')
pred_non_trivial = non_trivial_system.predict(X_test_SL)
mse_non_trivial = mean_squared_error(Y_test_SL, pred_non_trivial)
print('Test MSE of Baseline non-trivial system: ', mse_non_trivial)
print('Test R^2 of Baseline non-trivial system: ', r2_score(Y_test_SL, pred_non_trivial))

# model 1: l1 lasso
print('\n')
print('=== Model 1: Lasso Regression (Linear Regression with l1 regularization) ===')
# test
pred_best_lasso_test = best_lasso.predict(X_test_SL)
print('Test MSE of Lasso best model is: ', mean_squared_error(Y_test_SL, pred_best_lasso_test))
print('Test R^2 of Lasso best model is: ', r2_score(Y_test_SL, pred_best_lasso_test))

# model 2: l2 ridge
print('\n')
print('=== Model 2: Ridge Regression (Linear Regression with l2 regularization) ===')
pred_best_ridge_test = best_ridge.predict(X_test_SL)
print('Test MSE of Ridge best model is: ', mean_squared_error(Y_test_SL, pred_best_ridge_test))
print('Test R^2 of Ridge best model is: ', r2_score(Y_test_SL, pred_best_ridge_test))

# model 3: CART
print('\n')
print('=== Model 3: CART ===')
pred_best_cart_test = best_cart.predict(X_test_SL)
print('Test MSE of CART best model is: ', mean_squared_error(Y_test_SL, pred_best_cart_test))
print('Test R^2 of CART best model is: ', r2_score(Y_test_SL, pred_best_cart_test))

# model 4: random forest
print('\n')
print('=== Model 4: Random Forest ===')
pred_best_rf_test = best_rf.predict(X_test_SL)
print('Test MSE of RF best model is: ', mean_squared_error(Y_test_SL, pred_best_rf_test))
print('Test R^2 of RF best model is: ', r2_score(Y_test_SL, pred_best_rf_test))

# model 5: Adaboost
print('\n')
print('=== Model 5: Adaboost ===')
pred_best_ada_test = best_ada.predict(X_test_SL)
print('Test MSE of AdaBoost best model is: ', mean_squared_error(Y_test_SL, pred_best_ada_test))
print('Test R^2 of AdaBoost best model is: ', r2_score(Y_test_SL, pred_best_ada_test))

# model 6: KNN regressor
print('\n')
print('=== Model 6: KNN Regressor ===')
pred_best_knn_test = best_knn.predict(X_test_SL)
print('Test MSE of KNN best model is: ', mean_squared_error(Y_test_SL, pred_best_knn_test))
print('Test R^2 of KNN best model is: ', r2_score(Y_test_SL, pred_best_knn_test))

# model 7: SVR
from sklearn.svm import SVR
print('\n')
print('=== Model 7: SVR ===')
pred_best_svr_test = best_svr.predict(X_test_SL)
print('Test MSE of svr best model is: ', mean_squared_error(Y_test_SL, pred_best_svr_test))
print('Test R^2 of svr best model is: ', r2_score(Y_test_SL, pred_best_svr_test))
print('number of support vector', best_svr.n_support_)

print('\n')
print('================================ Transfer Learning Part ================================')

# modify the default parameters of np.load
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
Xt_test = np.load('data/X_test_TL.npy')
yt_test = np.load('data/Y_test_TL.npy')
np.load = np_load_old

filename8 = 'models/best_TL.sav'
best_tr = pickle.load(open(filename8, 'rb'))
filename9 = 'models/best_TL_baseline.sav'
baseline = pickle.load(open(filename9, 'rb'))

yt_test_pred = best_tr.predict(Xt_test)

# comparison with AdaBoost Regressor
yt_test_pred_base = baseline.predict(Xt_test)
print('TL Model 0: AdaBoost target test MSE:', mean_squared_error(yt_test, yt_test_pred_base))
print('TL Model 1: TrAdaBoost target test MSE for best model:', mean_squared_error(yt_test, yt_test_pred))
print('\n')

print('================================ Semi-Supervised Learning Part ================================')
# modify the default parameters of np.load
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
X_test_ssl_st = np.load('data/X_test_SSL.npy')
Y_test_ssl_st = np.load('data/Y_test_SSL.npy')
X_l_ssl_st = np.load('data/X_l_SSL.npy')
Y_l_ssl_st = np.load('data/Y_l_SSL.npy')
X_u_ssl_st = np.load('data/X_u_SSL.npy')
np.load = np_load_old


# load baseline knn model
filename10 = 'models/best_baseline_ssl.sav'
best_baseline_ssl = pickle.load(open(filename10, 'rb'))

# baseline knn
pred_best_baseline_test_ssl = best_baseline_ssl.predict(X_test_ssl_st)
print('SSL Model 0: KNN baseline test MSE = ', mean_squared_error(Y_test_ssl_st, pred_best_baseline_test_ssl))

# load cotraining models
filename11 = 'models/coregmodel1.sav'
coregmodel1 = pickle.load(open(filename11, 'rb'))
filename12 = 'models/coregmodel2.sav'
coregmodel2 = pickle.load(open(filename12, 'rb'))

# cotraining result
coreg_model_pred = (coregmodel1.predict(X_test_ssl_st) + coregmodel2.predict(X_test_ssl_st))/2
print('SSL Model 1: Cotraining regression based on KNN test MSE = ', mean_squared_error(Y_test_ssl_st, coreg_model_pred))


# model 2: label propagation by KNN
test_loss,_ = label_propagation_regression(X_l_ssl_st, Y_l_ssl_st, X_u_ssl_st, X_test_ssl_st, Y_test_ssl_st, 4, 1, 0.3161538461538462)
print('SSL Model 2: Label Propagation test MSE = ', test_loss)


plt.ioff()
plt.show()
