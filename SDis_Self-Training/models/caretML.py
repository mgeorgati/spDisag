import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('TKAgg')
import seaborn as sns
from catboost import CatBoostRegressor, MultiRegressionCustomObjective
from mainFunctions.basic import createFolder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor

SEED = 42

def fitlm(X, y):
    mod = LinearRegression()
    mod = mod.fit(X, y)
    return mod

def fitsgdregressor(X, y, batchsize, lrate, epoch):
    mod = SGDRegressor(max_iter=epoch, alpha=0, learning_rate='constant', eta0=lrate, verbose=1)
    mod = mod.fit(X, y)
    return mod

def plot_feature_importance(ROOT_DIR, importance, names, model_type, casestudy, city):
    import json
    with open(ROOT_DIR + '/logs/{}_ancData_list.txt'.format(city)) as f:
        json_data = json.load(f)
    featuresList = json_data['files']
    featuresList = [x.split('/')[-1].split('.')[0] for x in featuresList]

    #-------- Plot an image for the importance of variables --------
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(featuresList)
    
    print(feature_importance)
    print(feature_names)
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + '_FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    createFolder(ROOT_DIR + "/Results/{1}/{0}_importance".format(model_type,city))
    plt.savefig(ROOT_DIR + "/Results/{2}/{0}_importance/{1}.png".format(model_type,casestudy,city), dpi=300, bbox_inches='tight')
    
def fitrf(X, y, casestudy, city, ROOT_DIR):
    # mod = RandomForestRegressor(n_estimators = 10, random_state=SEED) # n_estimators = 10
    mod = RandomForestRegressor(n_estimators = 2, random_state=SEED, criterion='squared_error') # absolute_error, 'squared_error'
    mod = mod.fit(X, y)
    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    importances = mod.feature_importances_
    plot_feature_importance(ROOT_DIR, importances, feature_names, 'aprf', casestudy, city)  
    
    return mod

def fitmulti(X, y):
    #-------- NOT WORKING --------
    wrapper = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED, criterion='squared_error'))# mod = RandomForestRegressor(n_estimators = 10, random_state=SEED) # n_estimators = 10
    mod = wrapper.fit(X, y)
    return mod

#-------- LOSS FUNCTION FOR AMSTERDAM --------
class MultiRMSEObjectiveA(MultiRegressionCustomObjective):
           
        def calc_ders_multi(self, approxes, targets, weight):  
                # Assume that there are 5 output quantities. 
                # The first three should be summed to get the total, and the last two form another group that should be summed to get the total.                           
                assert len(approxes) == len(targets)
                sum1g1 = np.sum(approxes[0:5])
                sum2g1 = np.sum(targets[0:5])
                sum1g2 = np.sum(approxes[-7:])
                sum2g2 = np.sum(targets[-7:])
                grad_g1 = np.sign(sum2g1-sum1g1)
                grad_g2 = np.sign(sum2g2-sum1g2)
                grad = []
                hess = [[0 for j in range(len(targets))] for i in range(len(targets))]
                for index in range(len(targets)):
                        der1 = ((targets[index] - approxes[index]) + grad_g1 + grad_g2) * weight
                        der2 = -weight
                        grad.append(der1)
                        hess[index][index] = der2
                return (grad, hess)

#-------- LOSS FUNCTION FOR COPENHAGEN --------
class MultiRMSEObjectiveB(MultiRegressionCustomObjective):
                
        def calc_ders_multi(self, approxes, targets, weight):  
                # Assume that there are 5 output quantities. 
                # The first three should be summed to get the total, and the last two form another group that should be summed to get the total.                           
                assert len(approxes) == len(targets)
                sum1g1 = np.sum(approxes[0:5])
                sum2g1 = np.sum(targets[0:5])
                sum1g2 = np.sum(approxes[5:-2])
                sum2g2 = np.sum(targets[5:-2])
                sum1g3 = np.sum(approxes[-3:])
                sum2g3 = np.sum(targets[-3:])
                grad_g1 = np.sign(sum2g1-sum1g1)
                grad_g2 = np.sign(sum2g2-sum1g2)
                grad_g3 = np.sign(sum2g3-sum1g3)
                grad = []
                hess = [[0 for j in range(len(targets))] for i in range(len(targets))]
                for index in range(len(targets)):
                        der1 = ((targets[index] - approxes[index]) + grad_g1 + grad_g2 + grad_g3) * weight
                        der2 = -weight
                        grad.append(der1)
                        hess[index][index] = der2
                return (grad, hess)
                                
def fitcatbr(X, y, casestudy,city, ROOT_DIR):
    if city == "ams":
        mod = CatBoostRegressor(iterations=5, learning_rate=0.1, depth=1, task_type='CPU', thread_count=-1, loss_function=MultiRMSEObjectiveA(), eval_metric='MultiRMSE') 
    if city == "cph":
        mod = CatBoostRegressor(iterations=5, learning_rate=0.1, depth=1, task_type='CPU', thread_count=-1, loss_function=MultiRMSEObjectiveB(), eval_metric='MultiRMSE') 
    mod = mod.fit(X, y)
    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    plot_feature_importance(ROOT_DIR, mod.get_feature_importance(), feature_names, 'apcatbr', casestudy, city)   
    return mod

def fitxgbtree(X, y):
    # gbm = xgb.XGBRegressor(seed=SEED)
    gbm = xgb.XGBRegressor(seed=SEED, eval_metric='mae')
    # g13 = {'colsample_bytree': [0.4, 0.5], 'n_estimators': [50, 75, 100], 'max_depth': [3, 5, 7]}
    # # g14 = {'colsample_bytree': [0.3, 0.4, 0.5], 'n_estimators': [50, 75, 100], 'max_depth': [3, 5, 7]}
    # mod = RandomizedSearchCV(param_distributions=g13, estimator=gbm,
    #                         scoring='neg_mean_squared_error', n_iter=5, cv=4,
    #                         verbose=1)
    # mod = mod.fit(X, y)

    mod = gbm.fit(X, y)
    return mod

def fitxgbtreeM(X, y):
    #-------- NOT WORKING --------
    gbm = xgb.XGBRegressor(seed=SEED, eval_metric='mae')
    mod = gbm.fit(X, y)
    return mod
   
def fit(X, y, p, method, batchsize, lrate, epoch, ROOT_DIR, casestudy,city):
    X = X.reshape((-1, X.shape[2]))
    y = np.ravel(y)
    print('DOES IT COME IN HERE?')
    relevantids = np.where(~np.isnan(y))[0]
    relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
    idsamples = relevantids[relevsamples]
    print('| --- Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

    # Stratified sampling
    # numstrats = 10
    # bins = np.linspace(0, np.max(y)+0.0000001, numstrats+1)
    # digitized = np.digitize(y, bins)
    # numsamp = round(len(relevantids) * p[0] / numstrats + 0.5)
    # samplevals = [np.random.choice(np.where(digitized == i)[0], min(numsamp, len(y[digitized == i])), replace=False) for
    #               i in range(1, len(bins))]
    # idsamples = [item for sublist in samplevals for item in sublist]
    # np.random.shuffle(idsamples)

    X = X[idsamples,:]
    y = y[idsamples]
    # y = np.log(1+y)

    if(method.endswith('lm')):
        print('|| Fit: Linear Model')
        return fitlm(X, y)
    if (method.endswith('sgdregressor')):
        print('|| Fit: SGD Regressor')
        #return fitsgdregressor(X, y, batchsize, lrate, epoch)
    elif(method.endswith('rf')):
        print('|| Fit: Random Forests')
        return fitrf(X, y, casestudy, city, ROOT_DIR)
    elif(method.endswith('xgbtree')):
        print('|| Fit: XGBTree')
        #return fitxgbtree(X, y)
    else:
        return None

def fitM(X, y, p, method, ROOT_DIR, casestudy, city, group_split):
    
    X = X.reshape((-1, X.shape[2]))
    y = y.reshape((-1, y.shape[2]))
    
    relevantids = np.where(~np.isnan(y))[0]
    relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
    idsamples = relevantids[relevsamples]
    print('| --- Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

    X = X[idsamples,:]
    #y = y[idsamples]
    y = y[idsamples,:]
    # y = np.log(1+y)

    if(method.endswith('lm')):
        print('|| Fit: Linear Model')
        return fitlm(X, y)
    if (method.endswith('sgdregressor')):
        print('|| Fit: SGD Regressor')
        #return fitsgdregressor(X, y, batchsize, lrate, epoch)
    elif(method.endswith('rf')):
        print('|| Fit: Random Forests')
        return fitrf(X, y, casestudy, city, ROOT_DIR)
    elif(method.endswith('mltr')):
        print('|| Fit: Multi Output Regressor')
        return fitmulti(X, y)
    elif(method.endswith('catbr')):
        print('|| Fit: CatBoostRegressor')
        return fitcatbr(X, y, casestudy, city, ROOT_DIR)
    elif(method.endswith('xgbtree')):
        print('|| Fit: XGBTree')
        #return fitxgbtreeM(X, y)
    else:
        return None

def predictM(mod, X, attr_value):
    newX = X.reshape((-1, X.shape[2]))
    pred = mod.predict(newX)
    # pred = np.exp(pred)-1
    pred = pred.reshape(X.shape[0], X.shape[1], len(attr_value)) #HER IT NEED TO PASS len(attr_value)
    predlist = np.dsplit(pred, len(attr_value))
    return predlist #[pred]
