
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, MaxPooling3D, MaxPooling2D, \
    BatchNormalization, ZeroPadding2D, Cropping2D, ZeroPadding3D, Cropping3D, Dropout, UpSampling2D, Concatenate, \
    GlobalAveragePooling2D, Dense, UpSampling3D
from tensorflow.keras.layers import Input, concatenate, add, Dropout, Activation, PReLU, ReLU, LeakyReLU, Softmax, Layer
from tensorflow.keras.models import Model

#from keras.callbacks import TensorBoard
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import Callback

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
import datetime
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam


from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy

#from base.cnn_training.base import Network, NetworkTrainer
from functools import partial
import numpy as np
K.set_image_data_format('channels_last')  



class KerasNetwork():

    def __init__(self, network_builder, hyperparameters):
        self.network_builder = network_builder
        self.hyperparameters = hyperparameters
        self.model = self.network_builder.build()

    def save(self, path):
        self.model.save(path)

        

class OriginalUNETArchitecture3DKerasNetworkBuilder:

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build(self):
        shapes = self.hyperparams['image_size']
        n_labels = self.hyperparams['num_classes']
        inputs = Input((shapes[0], shapes[1], shapes[2], 1))

        conv2 = Conv3D(16, (5, 5, 5), padding='same')(inputs)
        act2_1 = PReLU()(conv2)
        fuse2 = add([inputs, act2_1])
        down2 = Conv3D(32, (2, 2, 2), strides=(1, 2, 2), padding='same')(fuse2)
        act2 = PReLU()(down2)

        conv3 = Conv3D(32, (5, 5, 5), padding='same')(act2)
        act3_1 = PReLU()(conv3)
        conv3_1 = Conv3D(32, (5, 5, 5), padding='same')(act3_1)
        act3_2 = PReLU()(conv3_1)
        fuse3 = add([act2, act3_2])
        down3 = Conv3D(64, (2, 2, 2), strides=(1, 2, 2), padding='same')(fuse3)
        act3 = PReLU()(down3)

        conv4 = Conv3D(64, (5, 5, 5), padding='same')(act3)
        act4_1 = PReLU()(conv4)
        conv4_1 = Conv3D(64, (5, 5, 5), padding='same')(act4_1)
        act4_2 = PReLU()(conv4_1)
        conv4_2 = Conv3D(64, (5, 5, 5), padding='same')(act4_2)
        act4_3 = PReLU()(conv4_2)
        fuse4 = add([act3, act4_3])
        down4 = Conv3D(128, (2, 2, 2), strides=(1, 2, 2), padding='same')(fuse4)
        act4 = PReLU()(down4)

        conv5 = Conv3D(128, (5, 5, 5), padding='same')(act4)
        act5_1 = PReLU()(conv5)
        conv5_1 = Conv3D(128, (5, 5, 5), padding='same')(act5_1)
        act5_2 = PReLU()(conv5_1)
        conv5_2 = Conv3D(128, (5, 5, 5), padding='same')(act5_2)
        act5_3 = PReLU()(conv5_2)
        fuse5 = add([act4, act5_3])
        down5 = Conv3D(256, (2, 2, 2), strides=(1, 2, 2), padding='same')(fuse5)
        act5 = PReLU()(down5)

        conv6 = Conv3D(256, (5, 5, 5), padding='same')(act5)
        act6_1 = PReLU()(conv6)
        conv6_1 = Conv3D(256, (5, 5, 5), padding='same')(act6_1)
        act6_2 = PReLU()(conv6_1)
        conv6_2 = Conv3D(256, (5, 5, 5), padding='same')(act6_2)
        act6_3 = PReLU()(conv6_2)
        fuse6 = add([act5, act6_3])

        up7 = Conv3DTranspose(256, (2, 2, 2), strides=(1, 2, 2), padding='same')(fuse6)
        act7_1 = PReLU()(up7)
        conc7 = concatenate([act7_1, fuse5], axis=4)
        conv7_1 = Conv3D(256, (5, 5, 5), padding='same')(conc7)
        act7_2 = PReLU()(conv7_1)
        conv7_2 = Conv3D(256, (5, 5, 5), padding='same')(act7_2)
        act7_3 = PReLU()(conv7_2)
        conv7_3 = Conv3D(256, (5, 5, 5), padding='same')(act7_3)
        act7_4 = PReLU()(conv7_3)
        conv7_4 = add([up7, act7_4])

        up8 = Conv3DTranspose(128, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv7_4)
        act8_1 = PReLU()(up8)
        conc8 = concatenate([act8_1, fuse4], axis=4)
        conv8_1 = Conv3D(128, (5, 5, 5), padding='same')(conc8)
        act8_2 = PReLU()(conv8_1)
        conv8_2 = Conv3D(128, (5, 5, 5), padding='same')(act8_2)
        act8_3 = PReLU()(conv8_2)
        conv8_3 = Conv3D(128, (5, 5, 5), padding='same')(act8_3)
        act8_4 = PReLU()(conv8_3)
        conv8_4 = add([up8, act8_4])

        up9 = Conv3DTranspose(64, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv8_4)
        act9_1 = PReLU()(up9)
        conc9 = concatenate([act9_1, fuse3], axis=4)
        conv9_1 = Conv3D(64, (5, 5, 5), padding='same')(conc9)
        act9_1 = PReLU()(conv9_1)
        conv9_2 = Conv3D(64, (5, 5, 5), padding='same')(act9_1)
        act9_2 = PReLU()(conv9_2)
        conv9_3 = add([up9, act9_2])

        up10 = Conv3DTranspose(32, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv9_3)
        act10 = PReLU()(up10)
        conc10 = concatenate([act10, fuse2], axis=4)
        conv10_1 = Conv3D(32, (5, 5, 5), padding='same')(conc10)
        act10_1 = PReLU()(conv10_1)
        conv10_2 = add([up10, act10_1])

        conv11 = Conv3D(n_labels, (1, 1, 1), activation='softmax')(conv10_2)

        model = Model(inputs=[inputs], outputs=[conv11])

        return model


