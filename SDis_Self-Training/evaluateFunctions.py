from sklearn import metrics
import numpy as np
from numpy import NaN, inf

def mae_error(actual, predicted):
    #predicted[(np.where((predicted <= -100000)))] = np.nan
    #predicted = np.nan_to_num(predicted, nan=0)    
    diff = actual - predicted
    mae = round(metrics.mean_absolute_error(actual, predicted),2)
    std = round(np.std(diff),2)
    return mae, diff, std

def rmse_error(actual, predicted):
    rmse = round(np.sqrt(metrics.mean_squared_error(actual, predicted)),2)
    return rmse

def mape_error(actual,predicted):
    actual[(np.where((actual >= 0) & (actual <= 1)))] = 1
    predicted[(np.where((predicted >= 0) & (predicted <= 1)))] = 1
    mape = round(metrics.mean_absolute_percentage_error(actual,predicted),2)
    return mape

def percentage_error(actual, predicted):
    #predicted= np.nan_to_num(predicted, nan=0, posinf=999999, neginf= -999999)
    #where_0A = np.where(0 < actual && actual <= 0)
    #where_0P = np.where(0 <predicted <= 0)
    actual[(np.where((actual >= 0) & (actual <= 1)))] = 1
    predicted[(np.where((predicted >= 0) & (predicted <= 1)))] = 1
    error = actual - predicted
    # Anything that is close to 0 is very wrong
    quotientArray = np.divide(error, actual) * 100 
    #quotientArray = np.divide(predicted, actual, out=np.ones_like(predicted), where=predicted==actual==0) * 100 
    quotient = round(np.nanmean(quotientArray),6)
    return round(quotient,2), quotientArray 



def prop_error(actual, predicted):
    #predicted= np.nan_to_num(predicted, nan=0, posinf=999999, neginf= -999999)
    diff = actual - predicted
    SD = np.sum(actual)
    PrE = (np.divide((diff * actual), SD)*1000)
    #divide calculation--> anywhere 'where' b does not equal zero. When b does equal zero, then it remains unchanged from whatever value you originally gave it in the 'out' argument
    return round(np.nanmean(PrE),4), PrE

def nrmse_error(actual, predicted):
    #predicted= np.nan_to_num(predicted, nan=0, posinf=999999, neginf= -999999)
    range = actual.max() - actual.min()
    return (np.sqrt(metrics.mean_squared_error(actual, predicted)))/range

def nmae_error(actual, predicted):
    #predicted= np.nan_to_num(predicted, nan=0, posinf=999999, neginf= -999999)
    range = actual.max() - actual.min()
    return (metrics.mean_absolute_error(actual, predicted))/range



