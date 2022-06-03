import numpy as np

def polygonValuesByID(ds, ids):
    
    uniqueids = np.unique(ids[~np.isnan(ids)])
    polygonvalues = {}
    for polid in uniqueids:
        polygonvalues[polid] = ds[ids == polid][0] #1 #### <<<< ----- CHANGED ----- >>>> ####

    return polygonvalues

def statsByID(ds, ids, stat='sum'):
    unique, counts = np.unique(np.unique(ids[~np.isnan(ids)]), return_counts=True)
    counts = dict(zip(unique, counts))

    stats = {}
    for polid in counts:
        if stat == 'sum':
            stats[polid] = np.nansum(ds[ids == polid])
        else:
            print('Invalid statistic')

    return stats

def extenddataset(X, y, ylr=np.array(None), transf=None):
    if transf == '2T6':
        auxtransf = np.random.randint(0, 5, X.shape[0])
        newX = np.concatenate((X,
                               np.flip(X[auxtransf == 0, :, :, :], 1),
                               np.flip(X[auxtransf == 1, :, :, :], 2),
                               np.flip(X[auxtransf == 2, :, :, :], (1,2)),
                               np.transpose(X[auxtransf == 3, :, :, :], (0, 2, 1, 3)),
                               np.rot90(X[auxtransf == 4, :, :, :], axes=(1,2)),
                               np.rot90(X[auxtransf == 5, :, :, :], axes=(2,1))), axis=0)
        newy = np.concatenate((y,
                               y[auxtransf == 0],
                               y[auxtransf == 1],
                               y[auxtransf == 2],
                               y[auxtransf == 3],
                               y[auxtransf == 4],
                               y[auxtransf == 5]))
        if ylr.any(): newylr = np.concatenate((ylr,
                                               ylr[auxtransf == 0],
                                               ylr[auxtransf == 1],
                                               ylr[auxtransf == 2],
                                               ylr[auxtransf == 3],
                                               ylr[auxtransf == 4],
                                               ylr[auxtransf == 5]))

        # auxtransfpos = np.random.choice(X.shape[0], round(X.shape[0]/2), replace=False)
        # auxtransfcat = np.random.randint(0, 5, round(X.shape[0]/2))
        # auxtransf = np.empty((X.shape[0]), dtype=int)
        # auxtransf[auxtransfpos] = auxtransfcat
        # newX = np.concatenate((newX,
        #                        np.flip(X[auxtransf == 0, :, :, :], 1),
        #                        np.flip(X[auxtransf == 1, :, :, :], 2),
        #                        np.flip(X[auxtransf == 2, :, :, :], (1,2)),
        #                        np.transpose(X[auxtransf == 4, :, :, :], (0, 2, 1, 3)),
        #                        np.rot90(X[auxtransf == 3, :, :, :], axes=(1,2)),
        #                        np.rot90(X[auxtransf == 5, :, :, :], axes=(2,1))), axis=0)
        # newy = np.concatenate((newy,
        #                        y[auxtransf == 0],
        #                        y[auxtransf == 1],
        #                        y[auxtransf == 2],
        #                        y[auxtransf == 3],
        #                        y[auxtransf == 4],
        #                        y[auxtransf == 5]))
        # if ylr.any(): newylr = np.concatenate((newylr,
        #                                        ylr[auxtransf == 0],
        #                                        ylr[auxtransf == 1],
        #                                        ylr[auxtransf == 2],
        #                                        ylr[auxtransf == 3],
        #                                        ylr[auxtransf == 4],
        #                                        ylr[auxtransf == 5]))

        sids = np.random.choice(newX.shape[0], newX.shape[0], replace=False)
        newX = newX[sids, :, :, :]
        newy = newy[sids]
        if ylr.any(): newylr = newylr[sids]

    else:
        newX = X
        newy = y
        newylr = ylr

    if ylr.any():
        return newX, newy, newylr
    else:
        return newX, newy

