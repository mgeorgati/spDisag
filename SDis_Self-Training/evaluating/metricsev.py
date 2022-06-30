import numpy as np
import sklearn.metrics as metrics

def mean_absolute_error(actual, predicted, areas):
    abserror = np.abs(np.array(actual) - np.array(predicted))
    areaweig = np.abs((1/np.array(areas)) / (sum(1/np.array(areas))))
    # areaweig = (areaweig ** 3)
    wmae = abserror * areaweig
    return np.sum(wmae) / np.sum(areaweig), wmae

def report_sdev_map(map):
    stdnormmean = np.nanstd(map / np.nanmean(map))
    return [stdnormmean]

def report_mae_y(y_true, y_pred):
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    maenormmean = mean_absolute_error / np.mean(y_true)
    return [maenormmean]

def report_rmse_y(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmsenormmean = np.sqrt(mse) / np.mean(y_true)
    return [rmsenormmean]

def report_r2_y(y_true, y_pred):
    r2 = metrics.r2_score(y_true, y_pred)
    r2normmean = r2 / np.mean(y_true)
    r2normitr = r2 / (np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    return [r2]

