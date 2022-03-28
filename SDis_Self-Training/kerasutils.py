import numpy as np, random
from sklearn import metrics
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.backend import *
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import *
from tensorflow.keras import activations
from itertools import product

from tensorflow.keras import backend as K
import tensorflow as tf

SEED = 42


def custom_activation(x):
    return K.elu(x)

####### THIS IS THE NEW CUSTOM LOSS FUNCTION FOR CNN
def averageinst(y_pred, i):
    y_pred[0] = tf.math.add(y_pred[0][:,:,:,i], y_pred[1][:,:,:,i])
    y_pred[0] = tf.math.divide(y_pred[0], tf.cast(2, tf.float32))
    return y_pred

def averagepreds(y_pred):
    i = tf.constant(0)
    y_pred = tf.while_loop(tf.less(i, len(y_pred[0][-1])), averageinst(y_pred, i)) # Does this work?
    return y_pred[0]

def custom_loss_fn(nsubgroups = [3, 2], nmodelpred = 2):
    def cl1(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true[:,:,:,0]))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        y_pred_final = tf.cond(nmodelpred == 2, averagepreds(y_pred), y_pred)

        # Custom loss sub-groups
        sumtrueg1 = tf.math.reduce_sum(y_true[:, :, :, 0:nsubgroups[0]])
        sumpredg1 = tf.math.reduce_sum(y_pred[:, :, :, 0:nsubgroups[0]])
        sumtrueg2 = tf.math.reduce_sum(y_true[:, :, :, nsubgroups[0]:nsubgroups[1]])
        sumpredg2 = tf.math.reduce_sum(y_pred[:, :, :, nsubgroups[0]:nsubgroups[1]])
        closssubgroups = tf.math.abs(sumtrueg1-sumpredg1) + tf.math.abs(sumtrueg2-sumpredg2)

        closs = closssubgroups + mean_squared_error(y_true, y_pred_final)
        return tf.math.divide_no_nan(closs, tf.cast(numfinite, tf.float32))
    return cl1
#######

def smoothL1(hubervalue = 0.5, stdivalue = 0.01):
    def sl1(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # # Huber Loss
        # HUBER_DELTA = hubervalue
        # x = K.abs(y_true - y_pred)
        # x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        #
        # huberloss = K.sum(x)
        # # huberloss = K.sum(x) - stdivalue * K.std(y_pred)
        # # huberloss = K.sum(x) + 10000*1/(1+K.std(y_pred))
        # return tf.math.divide_no_nan(huberloss, tf.cast(numfinite, tf.float32))

        # MAE
        # mae = K.mean(K.abs(y_pred - y_true))
        # mae = K.mean(K.abs(y_pred - y_true)) + 10000*1/(1+K.std(y_pred))
        # return tf.math.divide_no_nan(mae, tf.cast(numfinite, tf.float32))

        # # RMSE
        rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
        # rmse = K.sqrt(K.mean(K.square(y_pred - y_true))) + 10000*1/(1+K.std(y_pred))
        return tf.math.divide_no_nan(rmse, tf.cast(numfinite, tf.float32))
    return sl1


def smoothLC1(hubervalue = 0.5, stdivalue = 0.01):
    def sl1(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Version 2
        HUBER_DELTA = hubervalue
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)
        x1 = K.abs(y_true - y_pred_avg_two_channels)
        x1 = tf.where(x1 < HUBER_DELTA, 0.5 * x1 ** 2, HUBER_DELTA * (x1 - 0.5 * HUBER_DELTA))
        x2 = K.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1])
        x2 = tf.where(x2 < HUBER_DELTA, 0.5 * x2 ** 2, HUBER_DELTA * (x2 - 0.5 * HUBER_DELTA))
        sl = K.sum(x1) + K.sum(x2) + (10000*1/(1+tf.keras.backend.std(y_pred_avg_two_channels)))

        return tf.math.divide_no_nan(sl, tf.cast(numfinite, tf.float32))
    return sl1


def unet(inputs, filters=[2,4,8,16,32], dropout=0.5):
    conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(inputs)  # ALTERADO
    # conv1 = BatchNormalization()(conv1)  # ALTERADO
    # conv1 = Activation('relu')(conv1)  # ALTERADO

    conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # conv1 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv1)  # ALTERADO
    # conv1 = BatchNormalization()(conv1)  # ALTERADO
    # conv1 = Activation('relu')(conv1)  # ALTERADO

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(pool1)  # ALTERADO
    # conv2 = BatchNormalization()(conv2)  # ALTERADO
    # conv2 = Activation('relu')(conv2)  # ALTERADO

    conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer ='he_normal')(conv2)
    # conv2 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv2)  # ALTERADO
    # conv2 = BatchNormalization()(conv2)  # ALTERADO
    # conv2 = Activation('relu')(conv2)  # ALTERADO

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(pool2)  # ALTERADO
    # conv3 = BatchNormalization()(conv3)  # ALTERADO
    # conv3 = Activation('relu')(conv3)  # ALTERADO

    conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # conv3 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv3)  # ALTERADO
    # conv3 = BatchNormalization()(conv3)  # ALTERADO
    # conv3 = Activation('relu')(conv3)  # ALTERADO

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(pool3)  # ALTERADO
    # conv4 = BatchNormalization()(conv4)  # ALTERADO
    # conv4 = Activation('relu')(conv4)  # ALTERADO

    conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # conv4 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv4)  # ALTERADO
    # conv4 = BatchNormalization()(conv4)  # ALTERADO
    # conv4 = Activation('relu')(conv4)  # ALTERADO

    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(pool4)  # ALTERADO
    # conv5 = BatchNormalization()(conv5)  # ALTERADO
    # conv5 = Activation('relu')(conv5)  # ALTERADO

    conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # conv5 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv5)  # ALTERADO
    # conv5 = BatchNormalization()(conv5)  # ALTERADO
    # conv5 = Activation('relu')(conv5)  # ALTERADO

    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(filters[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(merge6)  # ALTERADO
    # conv6 = BatchNormalization()(conv6)  # ALTERADO
    # conv6 = Activation('relu')(conv6)  # ALTERADO

    conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # conv6 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv6)  # ALTERADO
    # conv6 = BatchNormalization()(conv6)  # ALTERADO
    # conv6 = Activation('relu')(conv6)  # ALTERADO

    up7 = Conv2D(filters[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(merge7)  # ALTERADO
    # conv7 = BatchNormalization()(conv7)  # ALTERADO
    # conv7 = Activation('relu')(conv7)  # ALTERADO

    conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # conv7 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv7)  # ALTERADO
    # conv7 = BatchNormalization()(conv7)  # ALTERADO
    # conv7 = Activation('relu')(conv7)  # ALTERADO

    up8 = Conv2D(filters[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(merge8)  # ALTERADO
    # conv8 = BatchNormalization()(conv8)  # ALTERADO
    # conv8 = Activation('relu')(conv8)  # ALTERADO

    conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # conv8 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv8)  # ALTERADO
    # conv8 = BatchNormalization()(conv8)  # ALTERADO
    # conv8 = Activation('relu')(conv8)  # ALTERADO

    up9 = Conv2D(filters[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(merge9)  # ALTERADO
    # conv9 = BatchNormalization()(conv9)  # ALTERADO
    # conv9 = Activation('relu')(conv9)  # ALTERADO

    conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(filters[0], 3, padding='same', kernel_initializer='he_normal')(conv9)  # ALTERADO
    # conv9 = BatchNormalization()(conv9)  # ALTERADO
    # conv9 = Activation('relu')(conv9)  # ALTERADO
    
    # New
    aux = concatenate([conv9, inputs], axis=3)
    # aux = conv9
    print("in UNET: conv9", conv9.shape, "aux:", aux.shape)
    output = Conv2D(12, 1, activation='linear')(aux) # CHANGED: Conv2D(1, 1, activation='linear')(aux)
    print("output of unet:", output.shape)
    return output


def compilecnnmodel(cnnmod, shape, lrate, dropout=0.5, filters=[2,4,8,16,32], lweights=[1/2, 1/2],
                    hubervalue=0.5, stdivalue=0.01):
    tf.random.set_seed(SEED)

    if cnnmod == 'cnnlm':
        shape = [7]
        mod = Sequential()
        mod.add(Dense(units=1, input_shape=shape))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'lenet':
        mod = Sequential()
        mod.add(Conv2D(filters=filters[0], kernel_size=3, padding='same', input_shape=shape, activation='relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Conv2D(filters=filters[1], kernel_size=3, padding='same'))
        mod.add(Activation('relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Dropout(rate=0.1))
        mod.add(Flatten())
        mod.add(Dense(units=filters[2]))
        mod.add(Activation('relu'))
        mod.add(Dense(units=1))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'vgg':
        mod = Sequential()
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', input_shape=shape, name='block1_conv1'))
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', name='block1_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv1'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv1'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv2'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        mod.add(Flatten(name='flatten'))
        mod.add(Dense(filters[4], activation='relu', name='fc1'))
        mod.add(Dense(filters[4], activation='relu', name='fc2'))
        mod.add(Dense(units=1, activation='linear', name='predictions'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'uenc':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        flat1 = Flatten()(drop5)
        dens1 = Dense(units=1, activation='linear', name='predictions')(flat1)
        mod = Model(inputs=inputs, outputs=dens1)
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'unet':
        
        inputs = Input(shape)
        print("input of the unet:", inputs.shape)
        # # Random transformation to apply
        # randomint = K.constant(random.randint(0, 5))
        #
        # def t0(inputs): return Lambda(lambda x: K.reverse(x, axes=1))(inputs) # Horizontal flip
        # def t1(inputs): return Lambda(lambda x: K.reverse(x, axes=2))(inputs) # Vertical flip
        # def t2(inputs): return Lambda(lambda x: K.reverse(x, axes=(1,2)))(inputs) # Horizontal and vertical flip
        # def t3(inputs): return Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(inputs) # Transpose
        # def t4(inputs): return Lambda(lambda x: K.reverse(t3(x), axes=1))(inputs) # Rotate clockwise
        # def t5(inputs): return Lambda(lambda x: K.reverse(t3(x), axes=2))(inputs) # Rotate counter clockwise
        # input_b = tf.case([(tf.equal(randomint, K.constant(0)), lambda: t0(inputs)),
        #                    (tf.equal(randomint, K.constant(1)), lambda: t1(inputs)),
        #                    (tf.equal(randomint, K.constant(2)), lambda: t2(inputs)),
        #                    (tf.equal(randomint, K.constant(3)), lambda: t3(inputs)),
        #                    (tf.equal(randomint, K.constant(4)), lambda: t4(inputs)),
        #                    (tf.equal(randomint, K.constant(5)), lambda: t5(inputs))],
        #                   exclusive=True)
        #
        # # Contrastive
        # # input_b = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(inputs)
        # processed_a = unet(inputs, filters, dropout)
        # processed_b = unet(input_b, filters, dropout)
        # # processed_b = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(processed_b)
        #
        # processed_b = tf.case([(tf.equal(randomint, K.constant(0)), lambda: t0(processed_b)),
        #                        (tf.equal(randomint, K.constant(1)), lambda: t1(processed_b)),
        #                        (tf.equal(randomint, K.constant(2)), lambda: t2(processed_b)),
        #                        (tf.equal(randomint, K.constant(3)), lambda: t3(processed_b)),
        #                        (tf.equal(randomint, K.constant(4)), lambda: t5(processed_b)),
        #                        (tf.equal(randomint, K.constant(5)), lambda: t4(processed_b))],
        #                      exclusive=True)

        result = unet(inputs, filters, dropout)
        resultModified = [result[:, :, :, 0:5], result[:, :, :, -7:]]
        print("EDO", len(resultModified), resultModified[0].shape)
        # result = Concatenate()([processed_a, processed_b])
        mod = Model(inputs=inputs, outputs=result)
        print("MOD", mod)
        sl1 = custom_loss_fn(nsubgroups = [5, 7], nmodelpred = 1)
        #sl1 = smoothL1(hubervalue=hubervalue, stdivalue=stdivalue)
        mod.compile(loss=sl1, optimizer=optimizers.Adam(lr=lrate))

        # Version 2, c)
        # slc1 = smoothLC1(hubervalue=hubervalue, stdivalue=stdivalue)
        # mod.compile(loss=slc1, optimizer=optimizers.Adam(lr=lrate))


    elif cnnmod == '2runet':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)


        # High resolution output
        outputhr = Conv2D(1, 1, activation='linear', name="highres")(conv9)

        # Sum of high resolution output
        avgpoolinghr = AveragePooling2D(pool_size=4)(outputhr)
        outputlr = Lambda(lambda x: x * 4, name="lowres")(avgpoolinghr)

        mod = Model(inputs=inputs, outputs=[outputhr, outputlr])
        mod.compile(loss=['mean_squared_error', 'mean_squared_error'],
                    loss_weights=lweights,
                    optimizer=optimizers.Adam(lr=lrate))

    return mod



def createpatches(X, patchsize, padding, stride=1, cstudy=None):
    if cstudy:
        try:
            fp = np.memmap(cstudy + '.dat', mode='r')
            print('Found .dat file')
            ninstances = int(fp.shape[0] / patchsize / patchsize / X.shape[2] / 4) # Divide by dimensions
            shapemmap = (ninstances, patchsize, patchsize, X.shape[2])
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='r', shape=shapemmap)
        except:
            print('Did not find .dat file')
            if padding:
                rowpad = int((patchsize - 1) / 2)
                colpad = int(round((patchsize - 1) / 2))
                newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
            else:
                newX = X
            newX[np.isnan(newX)] = -9999999
            patches = extract_patches_2d(newX, [16, 16])
            patches[patches == -9999999] = np.nan
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='w+', shape=patches.shape)
            fp[:] = patches[:]
            fp = fp.reshape(-1, patchsize, patchsize, X.shape[2])
        print("fp in ku:", fp.shape)
        return fp
    else:
        if padding:
            rowpad = int((patchsize - 1) / 2)
            colpad = int(round((patchsize - 1) / 2))
            newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            newX = X

        newX[np.isnan(newX)] = -9999999
        patches = extract_patches_2d(newX, [16, 16])
        patches[patches == -9999999] = np.nan
        patches = patches.reshape(-1, patchsize, patchsize, X.shape[2])
        print("patches in ku:", patches.shape)
        return patches

def reconstructpatches(patches, image_size, stride):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    mean = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / stride + 1)
    n_w = int((i_w - p_w) / stride + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        patch_count[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += ~np.isnan(p)
        ctignore = np.isnan(p)
        p[ctignore] = 0
        mean[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
        p[ctignore] = np.nan
    mean = np.divide(mean, patch_count, out=np.zeros_like(mean), where=patch_count != 0)

    # # Including variance
    # variance = np.zeros(image_size)
    # for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
    #     ctignore = np.isnan(p)
    #     p[ctignore] = 0
    #     variance[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += (p - mean[i * stride:i * stride + p_h, j * stride:j * stride + p_w]) ** 2
    #     p[ctignore] = np.nan
    # variance = np.divide(variance, patch_count, out=np.zeros_like(variance), where=patch_count != 0)
    print("mean in Reconstructs:", mean.shape)
    return mean
    # return [mean, variance]