import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

from models import robustLoss as rloss
from models import kerasutils as ku

def smoothLC1Adapt(aux_l1,aux_l2, aux_l3, aux_l4,aux_l5, aux_l6, aux_l7,aux_l8, aux_l9, aux_l10, 
            aux_l11, aux_l12, aux_l13, aux_l14, nmodelpred, nsubgroups=[5,12]):
    def slc1adapt(y_true, y_pred):

        if nmodelpred == 2: 
            y_pred_final = ku.averageinst(y_pred)
            y_pred = y_pred_final
        else: 
            y_pred = y_pred

        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)
        
        # Custom loss sub-groups
        sumtrueg1 = tf.math.reduce_sum(y_true[:, :, :, 0:nsubgroups[0]], axis=3, keepdims=True)
        sumpredg1 = tf.math.reduce_sum(y_pred[:, :, :, 0:nsubgroups[0]], axis=3, keepdims=True)
        sumtrueg2 = tf.math.reduce_sum(y_true[:, :, :, nsubgroups[0]:nsubgroups[1]], axis=3, keepdims=True)
        sumpredg2 = tf.math.reduce_sum(y_pred[:, :, :, nsubgroups[0]:nsubgroups[1]], axis=3, keepdims=True)

        x1 = aux_l1(sumtrueg1 - sumpredg1)
        loss1 = tf.math.divide_no_nan(K.sum(x1), tf.cast(numfinite, tf.float32))
        x2 = aux_l2(sumtrueg2 - sumpredg2)
        loss2 = tf.math.divide_no_nan(K.sum(x2), tf.cast(numfinite, tf.float32))
        print('----IN RBLF----')
        
        # DIFFERENCE BETWEEN PRED AND TRUE
        x3 = aux_l3(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
        print(x3)
        loss3 = tf.math.divide_no_nan(K.sum(x3), tf.cast(numfinite, tf.float32))
        print('---',loss3)
        x4 = aux_l4(y_true[:, :, :, 1] - y_pred[:, :, :, 1])
        loss4 = tf.math.divide_no_nan(K.sum(x4), tf.cast(numfinite, tf.float32))
        x5 = aux_l5(y_true[:, :, :, 2] - y_pred[:, :, :, 2])
        loss5 = tf.math.divide_no_nan(K.sum(x5), tf.cast(numfinite, tf.float32))
        x6 = aux_l6(y_true[:, :, :, 3] - y_pred[:, :, :, 3])
        loss6 = tf.math.divide_no_nan(K.sum(x6), tf.cast(numfinite, tf.float32))
        x7 = aux_l7(y_true[:, :, :, 4] - y_pred[:, :, :, 4])
        loss7 = tf.math.divide_no_nan(K.sum(x7), tf.cast(numfinite, tf.float32))
        x8 = aux_l8(y_true[:, :, :, 5] - y_pred[:, :, :, 5])
        loss8 = tf.math.divide_no_nan(K.sum(x8), tf.cast(numfinite, tf.float32))
        x9 = aux_l9(y_true[:, :, :, 6] - y_pred[:, :, :, 6])
        loss9 = tf.math.divide_no_nan(K.sum(x9), tf.cast(numfinite, tf.float32))
        x10 = aux_l10(y_true[:, :, :, 7] - y_pred[:, :, :, 7])
        loss10 = tf.math.divide_no_nan(K.sum(x10), tf.cast(numfinite, tf.float32))
        x11 = aux_l11(y_true[:, :, :, 8] - y_pred[:, :, :, 8])
        loss11 = tf.math.divide_no_nan(K.sum(x11), tf.cast(numfinite, tf.float32))
        x12 = aux_l12(y_true[:, :, :, 9] - y_pred[:, :, :, 9])
        loss12 = tf.math.divide_no_nan(K.sum(x12), tf.cast(numfinite, tf.float32))
        x13 = aux_l13(y_true[:, :, :, 10] - y_pred[:, :, :, 10])
        loss13 = tf.math.divide_no_nan(K.sum(x13), tf.cast(numfinite, tf.float32))
        x14 = aux_l14(y_true[:, :, :, 11] - y_pred[:, :, :, 11])
        loss14 = tf.math.divide_no_nan(K.sum(x14), tf.cast(numfinite, tf.float32))
        
        sl = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 
        return sl
    return slc1adapt



class CustomModelAdaptive(tf.keras.Model):
    def __init__(self, inputs, outputs, nmodelpred):
        super(CustomModelAdaptive, self).__init__(inputs, outputs)
        self.aux_l1 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l2 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l3 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l4 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l5 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l6 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l7 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l8 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l9 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l10 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l11 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l12 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l13 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l14 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.sladapt = smoothLC1Adapt(self.aux_l1, self.aux_l2, self.aux_l3, self.aux_l4, self.aux_l5, self.aux_l6, 
                self.aux_l7, self.aux_l8, self.aux_l9, self.aux_l10, self.aux_l11, self.aux_l12, self.aux_l13, self.aux_l14, nmodelpred)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) # Must do
            loss = self.sladapt(y, y_pred)

        tf.print(loss)

        # Compute gradients
        model_vars = self.trainable_variables
        loss_vars = tf.unstack(self.aux_l1.trainable_variables + self.aux_l2.trainable_variables + self.aux_l3.trainable_variables 
                        + self.aux_l4.trainable_variables + self.aux_l5.trainable_variables + self.aux_l6.trainable_variables
                        + self.aux_l7.trainable_variables + self.aux_l8.trainable_variables + self.aux_l9.trainable_variables
                        + self.aux_l10.trainable_variables + self.aux_l11.trainable_variables + self.aux_l12.trainable_variables
                        + self.aux_l13.trainable_variables + self.aux_l14.trainable_variables)
        trainable_vars = list(model_vars) + list(loss_vars)
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def smoothLC1AdaptCPH(aux_l1,aux_l2, aux_l3, aux_l4,aux_l5, aux_l6, aux_l7,aux_l8, aux_l9, aux_l10, 
            aux_l11, aux_l12, aux_l13, aux_l14, aux_l15, nmodelpred, nsubgroups):
    def slc1adapt(y_true, y_pred):

        if nmodelpred == 2: 
            y_pred_final = ku.averageinst(y_pred)
            y_pred = y_pred_final
        else: 
            y_pred = y_pred

        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)
        
        # Custom loss sub-groups
        sumtrueg1 = tf.math.reduce_sum(y_true[:, :, :, 0:nsubgroups[0]], axis=3, keepdims=True)
        sumpredg1 = tf.math.reduce_sum(y_pred[:, :, :, 0:nsubgroups[0]], axis=3, keepdims=True)
        sumtrueg2 = tf.math.reduce_sum(y_true[:, :, :, nsubgroups[0]:nsubgroups[1]], axis=3, keepdims=True)
        sumpredg2 = tf.math.reduce_sum(y_pred[:, :, :, nsubgroups[0]:nsubgroups[1]], axis=3, keepdims=True)

        x1 = aux_l1(sumtrueg1 - sumpredg1)
        loss1 = tf.math.divide_no_nan(K.sum(x1), tf.cast(numfinite, tf.float32))
        x2 = aux_l2(sumtrueg2 - sumpredg2)
        loss2 = tf.math.divide_no_nan(K.sum(x2), tf.cast(numfinite, tf.float32))
        print('----IN RBLF----')
        
        # DIFFERENCE BETWEEN PRED AND TRUE
        x3 = aux_l3(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
        print(x3)
        loss3 = tf.math.divide_no_nan(K.sum(x3), tf.cast(numfinite, tf.float32))
        print('---',loss3)
        x4 = aux_l4(y_true[:, :, :, 1] - y_pred[:, :, :, 1])
        loss4 = tf.math.divide_no_nan(K.sum(x4), tf.cast(numfinite, tf.float32))
        x5 = aux_l5(y_true[:, :, :, 2] - y_pred[:, :, :, 2])
        loss5 = tf.math.divide_no_nan(K.sum(x5), tf.cast(numfinite, tf.float32))
        x6 = aux_l6(y_true[:, :, :, 3] - y_pred[:, :, :, 3])
        loss6 = tf.math.divide_no_nan(K.sum(x6), tf.cast(numfinite, tf.float32))
        x7 = aux_l7(y_true[:, :, :, 4] - y_pred[:, :, :, 4])
        loss7 = tf.math.divide_no_nan(K.sum(x7), tf.cast(numfinite, tf.float32))
        x8 = aux_l8(y_true[:, :, :, 5] - y_pred[:, :, :, 5])
        loss8 = tf.math.divide_no_nan(K.sum(x8), tf.cast(numfinite, tf.float32))
        x9 = aux_l9(y_true[:, :, :, 6] - y_pred[:, :, :, 6])
        loss9 = tf.math.divide_no_nan(K.sum(x9), tf.cast(numfinite, tf.float32))
        x10 = aux_l10(y_true[:, :, :, 7] - y_pred[:, :, :, 7])
        loss10 = tf.math.divide_no_nan(K.sum(x10), tf.cast(numfinite, tf.float32))
        x11 = aux_l11(y_true[:, :, :, 8] - y_pred[:, :, :, 8])
        loss11 = tf.math.divide_no_nan(K.sum(x11), tf.cast(numfinite, tf.float32))
        x12 = aux_l12(y_true[:, :, :, 9] - y_pred[:, :, :, 9])
        loss12 = tf.math.divide_no_nan(K.sum(x12), tf.cast(numfinite, tf.float32))
        x13 = aux_l13(y_true[:, :, :, 10] - y_pred[:, :, :, 10])
        loss13 = tf.math.divide_no_nan(K.sum(x13), tf.cast(numfinite, tf.float32))
        x14 = aux_l14(y_true[:, :, :, 11] - y_pred[:, :, :, 11])
        loss14 = tf.math.divide_no_nan(K.sum(x14), tf.cast(numfinite, tf.float32))
        x15 = aux_l15(y_true[:, :, :, 11] - y_pred[:, :, :, 11])
        loss15 = tf.math.divide_no_nan(K.sum(x15), tf.cast(numfinite, tf.float32))
        sl = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15
        return sl
    return slc1adapt

class CustomModelAdaptiveCPH(tf.keras.Model):
    def __init__(self, inputs, outputs, nmodelpred, nsubgroups):
        super(CustomModelAdaptiveCPH, self).__init__(inputs, outputs)
        self.aux_l1 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l2 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l3 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l4 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l5 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l6 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l7 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l8 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l9 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l10 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l11 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l12 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l13 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l14 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l15 = rloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.sladapt = smoothLC1AdaptCPH(self.aux_l1, self.aux_l2, self.aux_l3, self.aux_l4, self.aux_l5, self.aux_l6, 
                self.aux_l7, self.aux_l8, self.aux_l9, self.aux_l10, self.aux_l11, self.aux_l12, self.aux_l13, self.aux_l14, self.aux_l15, nmodelpred, nsubgroups)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) # Must do
            loss = self.sladapt(y, y_pred)

        tf.print(loss)

        # Compute gradients
        model_vars = self.trainable_variables
        loss_vars = tf.unstack(self.aux_l1.trainable_variables + self.aux_l2.trainable_variables + self.aux_l3.trainable_variables 
                        + self.aux_l4.trainable_variables + self.aux_l5.trainable_variables + self.aux_l6.trainable_variables
                        + self.aux_l7.trainable_variables + self.aux_l8.trainable_variables + self.aux_l9.trainable_variables
                        + self.aux_l10.trainable_variables + self.aux_l11.trainable_variables + self.aux_l12.trainable_variables
                        + self.aux_l13.trainable_variables + self.aux_l14.trainable_variables + self.aux_l15.trainable_variables)
        trainable_vars = list(model_vars) + list(loss_vars)
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}