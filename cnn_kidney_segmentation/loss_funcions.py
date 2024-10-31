import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
import numpy as np

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    targets = tf.cast(targets, dtype='float32')
    #inputs = tf.cast(inputs, dtype='float64')


    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice



def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                            K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                    K.floatx())


def GeneralWeightedDiceLoss(weights=np.array([0.2, 0.8])):
    '''
    dice multiclass
    '''
    def loss(y_true, y_pred):
        weighted_soft_Dice_loss = 0
        for i in range(len(weights)):
            weighted_soft_Dice_loss += dice_coefficient(y_true[:, :, :, :, i], y_pred[:, :, :, :, i]) * weights[i]

        loss = 1 - weighted_soft_Dice_loss
        return loss

    return loss



def WeightedCategoricalLoss( weights= np.array([0.2,0.8])):
    def weighted_categorical_crossentropy_loss(y_true, y_pred):
        # a veces hay que hacerle un cast, sino da error 
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return weighted_categorical_crossentropy_loss