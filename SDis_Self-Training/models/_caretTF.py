import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from utils import npu

from models import ku

SEED = 42


def get_callbacks():
    return [
        # ReduceLROnPlateau(monitor='loss', min_delta=0.0, patience=3, factor=0.1, min_lr=5e-6, verbose=1),
        # EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=1, restore_best_weights=True)
        EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=1, restore_best_weights=True)
    ]

class DataGenerator(utils.Sequence):
    'Generates data for Keras'

    def __init__(self, Xdataset, ydataset, idsamples, batch_size, p):
        'Initialization'
        self.X, self.y = Xdataset, ydataset
        self.batch_size = batch_size
        self.p = p
        self.idsamples = idsamples

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.idsamples) / self.batch_size)

    def __getitem__(self, idx):
        'Generate one batch of data'
        idsamplesbatch = self.idsamples[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[idsamplesbatch]
        batch_X[np.isnan(batch_X)] = 0 ## Alterado
        batch_y = self.y[idsamplesbatch]
        print("Generator output:", np.array(batch_X).shape, np.array(batch_y).shape)
        return np.array(batch_X), np.array(batch_y)


def fitcnn(X, y, p, ROOT_DIR, city, cnnmod, cnnobj, casestudy, epochs, batchsize, extdataset):
    tf.random.set_seed(SEED)

    # Reset model weights
    print('------- LOAD INITIAL WEIGHTS')
    cnnobj.load_weights(ROOT_DIR + '/Temp/{}/models_'.format(city) + casestudy + '.h5')

    if cnnmod == 'lenet':
        print('| --- Fit - 1 resolution Le-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            Xfit = X[idsamples, :, :, :]
            yfit = y[idsamples]
            hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize)]
            return hislist

    elif cnnmod == 'vgg':
        print('| --- Fit - 1 resolution VGG-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod == 'uenc':
        print('| --- Fit - 1 resolution U-Net encoder')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod.endswith('unet'):
        print(y.shape)
        # Compute midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        print(middcol, middrow)
        # Train only with patches having middpixel different from NaN
        relevantids = np.where(~np.isnan(y[:,middrow,middcol,0]))[0]
        print(relevantids.shape)
        # Train only with patches having finit values in all pixels
        # relevantids = np.where(~np.isnan(y).any(axis=(1,2,3)))[0]

        if len(p) > 1:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids)), replace=False)
        else:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]

        # Stratified sampling
        # numstrats = 10
        # sumaxis0 = np.sum(y, axis=(1, 2, 3))
        # bins = np.linspace(0, np.nanmax(sumaxis0)+0.0000001, numstrats+1)
        # digitized = np.digitize(sumaxis0, bins)
        # numsamp = round(len(relevantids) * p[0] / numstrats + 0.5)
        # samplevals = [np.random.choice(np.where(digitized == i)[0], min(numsamp, len(sumaxis0[digitized == i])), replace=False) for
        #               i in range(1, len(bins))]
        # idsamples = [item for sublist in samplevals for item in sublist]
        # np.random.shuffle(idsamples)

        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if(cnnmod == 'unet'):
            print("X shape UNET", X.shape)
            print("y shape UNET = demo", y.shape)
            if extdataset:
                #NOT DEFINED
                Xfit = X[idsamples, :, :, :]
                yfit = y[idsamples]
                yfit[np.isnan(yfit)] = 0
                newX, newy = npu.extenddataset(Xfit, yfit, transf=extdataset)
                hislist = [cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)]
                return hislist
            else:
                if len(p) > 1:
                    print('| --- Fit-generator - 1 resolution U-Net Model')
                    hislist = []
                    Xgen = X[idsamples, :, :, :]
                    ygen = y[idsamples]
                    for i in range(len(p)):
                        cnnobj.load_weights('models_' + casestudy + '.h5')
                        mod = cnnobj.fit(generator(Xgen, ygen, batchsize),
                                         steps_per_epoch=round(len(idsamples) * p[i] / batchsize), epochs=epochs)
                        hislist.append(mod)
                        cnnobj.save_weights('snapshot_' + casestudy + '_' + str(i) + '.h5')
                    return hislist
                else:
                    print('| --- Fit - 1 resolution U-Net Model')
                    # Xfit = X[idsamples, :, :, :] ## Comentado
                    # yfit = y[idsamples] ## Comentado
                    # ctignore = np.isnan(Xfit) ## Comentado
                    # Xfit[ctignore] = 0 ## Comentado

                    # yfit = np.log(1+yfit)
                    # yfit[np.isnan(yfit)] = 0
                    # hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize)]

                    training_generator = DataGenerator(X, y, idsamples, batch_size=batchsize, p=p[0])
                    hislist = [cnnobj.fit(training_generator, epochs=epochs)] #This is history object storing info, e.g. for loss at each epoch

                    # hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize, callbacks=get_callbacks())]
                    print("???? hislist")
                    test_type(hislist)
                    return hislist

        elif(cnnmod.startswith('2r')):
            print('| --- Fit - 2 resolution U-Net Model')
            if extdataset:
                newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            else:
                newX, newy = X[idsamples, :, :, :], y[idsamples]
            newylr = skimage.measure.block_reduce(newy, (1,4,4,1), np.sum) # Compute sum in 4x4 blocks
            newy = [newy, newylr]

            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
            # return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize, callbacks=get_callbacks())

    else:
        print('Fit CNN - Unknown model')

def predictloop(cnnmod, patches, batchsize):
    print("in predictloop, patches:", patches.shape)
    # Custom batched prediction loop
    final_shape = [patches.shape[0], patches.shape[1], patches.shape[2], 12] #This 12 needs to be changes based on input demo ----!!!----
    print("in predictloop, final_patches:", final_shape)
    y_pred_probs = np.empty(final_shape,
                            dtype=np.float32)  # pre-allocate required memory for array for efficiency

    # Array with first number for each batch
    batch_indices = np.arange(start=0, stop=patches.shape[0], step=batchsize)  # row indices of batches
    batch_indices = np.append(batch_indices, patches.shape[0])  # add final batch_end row

    i=1
    for index in np.arange(len(batch_indices) - 1):
        batch_start = batch_indices[index]  # first row of the batch
        batch_end = batch_indices[index + 1]  # last row of the batch

        # y_pred_probs[batch_start:batch_end] = cnnmod.predict_on_batch(patches[batch_start:batch_end])
        # y_pred_probs[batch_start:batch_end] = np.expand_dims(
        #     cnnmod.predict_on_batch(patches[batch_start:batch_end])[:,:,:,0], axis=3)

        # Alterado

        # Replace NANs in patches by zero
        ctignore = np.isnan(patches[batch_start:batch_end])
        patchespred = patches[batch_start:batch_end].copy()
        patchespred[ctignore] = 0
        # WE'LL NEED TO ADJUST THAT IF WE INCLUDE THE FLIPPED IMAGES 
        #y_pred_probs[batch_start:batch_end] = np.expand_dims(
            #np.mean(cnnmod.predict_on_batch(patchespred), axis=3), axis=3)
        y_pred_probs[batch_start:batch_end] = cnnmod.predict_on_batch(patchespred) #np.expand_dims(cnnmod.predict_on_batch(patchespred), axis=3) #cnnmod.predict_on_batch(patchespred)
        
        # Replace original NANs with NAN
        patchespred[ctignore] = np.nan

        i = i+1
        if(i%1000 == 0): print('»» Batch', i, '/', len(batch_indices), end='\r')
    
    return y_pred_probs

def channelSplit(image):
    return np.dsplit(image,image.shape[-1])

def predictcnn(obj, mod, fithistory, casestudy, ancpatches, dissshape, batchsize, stride=1):
    if mod == 'lenet':
        print('| --- Predicting new values, Le-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'vgg':
        print('| --- Predicting new values, VGG-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'uenc':
        print('| --- Predicting new values, U-Net encoder')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod.endswith('unet'):
        if(mod == 'unet'):
            if len(fithistory) > 1:
                print('| --- Predicting new patches from several models, 1 resolution U-Net')
                predhr = []
                for i in range(len(fithistory)):
                    obj.load_weights('snapshot_' + casestudy + '_' + str(i) + '.h5')
                    predhr.append(obj.predict(ancpatches, batch_size=batchsize, verbose=1))
                print('| ---- Reconstructing HR images from patches..')
                for i in range(len(predhr)): predhr[i] = ku.reconstructpatches(predhr[i], dissshape, stride)
                return predhr
            else:
                print('| --- Predicting new patches, 1 resolution U-Net')
                predhr = predictloop(obj, ancpatches, batchsize=batchsize)
                print("predhr_1")
                test_type(predhr)
                print('| ---- Reconstructing HR image from patches..')
                if len(predhr.shape) == 4:
                    aux = [ ku.reconstructpatches(predhr[:,:,:,a], (dissshape[0],dissshape[1]), stride) for a in range(predhr.shape[3]) ]
                    predlist = np.dsplit(np.moveaxis( np.array(aux), 0, 2), 12)
                    return predlist

        elif mod.startswith('2r'):
            print('| --- Predicting new patches, 2 resolution U-Net')
            predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)[0]
            print('| ---- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            return predhr

    else:
        print('Predict CNN - Unknown model')
