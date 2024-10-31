# enconding: utf-8
import random
from re import A
import SimpleITK as sitk
from SimpleITK.SimpleITK import GetPixelIDValueAsString, Or
import numpy as np
import shutil
from natsort import os_sorted
import logging, os, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import datetime
from os.path import dirname, join, abspath
import matplotlib.pyplot as plt
import cv2
import functools
import time
import math
from natsort import natsorted
import statistics
import pandas as pd
import csv


from tensorflow.python.keras.backend import binary_crossentropy
sys.path.insert(0, '..')
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.python.keras.metrics import Metric, binary_accuracy, categorical_accuracy
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, MaxPooling3D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Cropping2D, ZeroPadding3D, Cropping3D, Dropout, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Input, concatenate, add, Dropout, Activation, PReLU, ReLU, LeakyReLU, Softmax, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.test.gpu_device_name()



from preprocessing import preprocessing_tools
from augmentation import data_augmentation_tools
from Visualization.itk_visualization import sitk_show
from sklearn.model_selection import train_test_split
from dataset import dataset_generation_tools
from cnn_kidney_segmentation.segmentation_networks import OriginalUNETArchitecture3DKerasNetworkBuilder, KerasNetwork, ModifiedUNETBatchNormArchitecture3DKerasNetworkBuilder
from cnn_kidney_segmentation.loss_funcions import dice_coefficient, dice_coefficient_loss, custom_sparse_categorical_accuracy, GeneralWeightedDiceLoss, WeightedCategoricalLoss
from postprocessing.postprocessing_tools import otsu_thresholding
from augmentation.data_augmentation_tools import rotate, flip, translate, translate_scipy
from postprocessing import postprocessing_tools

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)



def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds./n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def remove_folder(path):
    # check if folder exists
    # remove if exists
    shutil.rmtree(path)

def divide_dataset(data_path, study, modality, select_pat = None, include_m0s = None):
    # Create folders to hold imgs and mask (train, val and test)
    folders = ['train', 'test']
    subfolders = ['train', 'val']
    subsubfolders = ['images', 'masks']

    if os.path.exists(data_path + 'temporal') == True:
        remove_folder(data_path + 'temporal')
        remove_folder(data_path + 'temporal/' + subsubfolders[0])
        remove_folder(data_path + 'temporal/' + subsubfolders[1])
    else:
        os.makedirs(data_path + 'temporal')
        os.makedirs(data_path + 'temporal/' + subsubfolders[0])
        os.makedirs(data_path + 'temporal/' + subsubfolders[1])

    for folder in folders:
        if os.path.exists(data_path + folder) == True:
            remove_folder(data_path + folder)
        else:
            os.makedirs(data_path + folder)

        if folder == 'train':
            for subfolder in subfolders:
                if os.path.exists(data_path + folder + '/' + subfolder) == True:
                    remove_folder(data_path + folder + '/' + subfolder)
                else:
                    os.makedirs(data_path + folder + '/' + subfolder)

                for subsubfolder in subsubfolders:
                    if os.path.exists(data_path + folder + '/' + subfolder + '/' + subsubfolder) == True:
                        remove_folder(data_path + folder + '/' + subfolder + '/' + subsubfolder)
                    else:
                        os.makedirs(data_path + folder + '/' + subfolder + '/' + subsubfolder)  
        else:
            for subsubfolder in subsubfolders:
                    if os.path.exists(data_path + folder  + '/' + subsubfolder) == True:
                        remove_folder(data_path + folder  + '/' + subsubfolder)
                    else:
                        os.makedirs(data_path + folder  + '/' + subsubfolder)  


    # leave one out method
    patient_list = os.listdir(im_path) # same as mask_ path patients
    
    
    if select_pat is not None:
        patient_list_new = patient_list
        a = (range(len(patient_list_new)))
        for pats in range(0,len(patient_list)):
            if patient_list_new[pats] == select_pat:
                test_ids = pats
                a = np.delete(a, [pats])
                break
                
        #patient_list = np.delete(patient_list[patient_list==test_ids])
        train_ids = np.random.permutation(a)
        #test_ids = patient_list[(patient_list==select_pat)]

        for i in range(len(train_ids)): print('Train patients:', patient_list[train_ids[i]])
        print('Test patients:', patient_list[test_ids])
    else:
        a = np.random.permutation(range(len(patient_list)))
        train_ids = a[0:len(patient_list)-1]
        test_ids = a[len(patient_list)-1]
        train_ids = np.array(train_ids)

        for i in range(len(train_ids)): print('Train patients:', patient_list[train_ids[i]])
        print('Test patients:', patient_list[test_ids])

    

    # Save TRAIN images
    if include_m0s is True:
        first_image = 0
    else:
        first_image = 1

    for pats in range(0, len(train_ids)):
        imgs_to_train = os_sorted(os.listdir(im_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/'))
        masks_to_train = os_sorted(os.listdir(mask_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/'))


        for imgs in range(first_image, len(imgs_to_train)):
            img_to_train = sitk.ReadImage(im_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/' + imgs_to_train[imgs])
            mask_to_train = sitk.ReadImage(mask_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/' + masks_to_train[imgs])
            mask_to_train = preprocessing_tools.fill_holes(mask_to_train)
            img_name = patient_list[train_ids[pats]] + '_' + study + '_' + modality + '_' + str(imgs+1) + '.nii'
            sitk.WriteImage(img_to_train, '//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/images/' + img_name )
            mask_name = patient_list[train_ids[pats]] + '_' + study + '_' + modality + '_' + str(imgs+1) + '.mha'
            sitk.WriteImage(mask_to_train, '//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/masks/' + mask_name)

    temporal_root = '//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/images'
    allTrainFileNames = os.listdir(temporal_root)
    random.shuffle(allTrainFileNames)

    val_ratio = 0.15

    train_FileNames, val_FileNames = np.split(np.array(allTrainFileNames), [int(len(allTrainFileNames) * (1 - val_ratio))])

    for divided_train_imgs in train_FileNames:
        filename, _ = os.path.splitext(divided_train_imgs)

        img = sitk.ReadImage('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/images/' + filename + '.nii')
        mask = sitk.ReadImage('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/masks/' + filename + '.mha')


        sitk.WriteImage(img, ('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/train/train/images/' + filename + '.nii'))
        sitk.WriteImage(mask, ('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/train/train/masks/' + filename + '.mha'))

    for divided_val_imgs in val_FileNames:
        filename, _ = os.path.splitext(divided_val_imgs)

        img = sitk.ReadImage('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/images/' + filename + '.nii')
        mask = sitk.ReadImage('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/temporal/masks/' + filename + '.mha')

        sitk.WriteImage(img, ('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/train/val/images/' + filename + '.nii'))
        sitk.WriteImage(mask, ('//172.18.93.155/aoyarzun/RM_RENAL/DATASET/train/val/masks/' + filename + '.mha'))


    # Save TEST images
    imgs_to_test = os_sorted(os.listdir(im_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/'))
    masks_to_test = os_sorted(os.listdir(mask_path + patient_list[train_ids[pats]] + '/' + study + '/' + modality + '/'))

    for imgs in range(first_image, len(imgs_to_test)):
        img_to_test = sitk.ReadImage(im_path + patient_list[test_ids] + '/' + study + '/' + modality + '/' + imgs_to_test[imgs])
        mask_to_test = sitk.ReadImage(mask_path + patient_list[test_ids] + '/' + study + '/' + modality + '/' + masks_to_test[imgs])
        mask_to_test = preprocessing_tools.fill_holes(mask_to_test)

        img_name = patient_list[test_ids] + '_' + study + '_' + modality + '_' + str(imgs+1) + '.nii'
        sitk.WriteImage(img_to_test, '//172.18.93.155/aoyarzun/RM_RENAL/DATASET/test/images/' + img_name )
        mask_name = patient_list[test_ids] + '_' + study + '_' + modality + '_' + str(imgs+1) + '.mha'
        sitk.WriteImage(mask_to_test, '//172.18.93.155/aoyarzun/RM_RENAL/DATASET/test/masks/' + mask_name)

    remove_folder(data_path + 'temporal')
        


class DataGenerator(tf.keras.utils.Sequence):
    """ Defining DataGenerator

    images_paths - list of paths to input images
    labels_paths - list of paths to label images
    batch_size - int batch size
    image_dimensions - image will be resized to provided dimensions
    shuffle - shuffle indices on epoch end
    augment - augment data by images*aug_factor
    aug_factor - aug_factor - 1 different augmentations (up to max. 8)
    flip - align object of interest to left side of image
    sharpen - apply UnsharpMask filter to image to sharpen edges
    """
    def __init__(self, images_paths, labels_paths, n_classes, batch_size=16, image_dimensions=[96, 96, 3],
                 shuffle=False, augment=False, aug_factor=8, flip=False, sharpen=False, intense_window=False, otsu=True):
        self.labels_paths = labels_paths  # array of labels
        self.images_paths = images_paths  # array of image paths
        self.n_class = n_classes
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.shuffle = shuffle  # shuffle bool
        self.augment = augment  # augment data bool
        self.aug_factor = aug_factor
        self.flip = flip
        self.sharpen = sharpen
        self.IW = intense_window
        self.otsu= otsu
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(os.listdir(self.images_paths)) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(os.listdir(self.images_paths)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # select data and load images

        images, labels = data_preprocess(self.images_paths, self.labels_paths, self.n_class, indexes,
                                         self.dim, self.augment, self.aug_factor, self.flip, self.sharpen, self.IW, self.otsu)
        # # test prints
        # random_idx = np.random.randint(0, images.shape[0], 5)
        # for i in random_idx:
        #     image = images[i, :, :, 0]
        #     itk_visualization.sitk_show(sitk.GetImageFromArray(image), "slice")
        #     if labels is not None:
        #         label = labels[i, :, :, 2]
        #         itk_visualization.sitk_show(sitk.GetImageFromArray(label.astype(np.uint8)), "label")
        #         input('Press Enter..')
        #         # pink = [255, 105, 180]
        #         # green = [0, 255, 0]
        #         # gold = [255, 215, 0]
        #         overlay = sitk.LabelOverlay(sitk.GetImageFromArray(image), sitk.GetImageFromArray(label.astype(np.uint8)),
        #                                     opacity=0.5)
        #         itk_visualization.sitk_show(overlay, "overlaid")
        #     input('Press Enter Label..')
        return images, labels

def histo_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(50,50))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        #img_to_eq = imgs[i]
        #img_equalized[i] = clahe.apply(np.array(imgs[i], dtype = np.uint8))
        img_equalized = clahe.apply(imgs[i])
        imgs_equalized.append(img_equalized[:, :, :, np.newaxis])

    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def crop_images(input_image, input_mask):
    
    cropped_img = np.empty((3,96,96))
    #input_image = sitk.Cast(input_image, sitk.sitkInt16)
    for i in range(0, 3):
        mask_s = input_mask[:,:,i]
        #print(mask_s.GetPixelIDTypeAsString())
        # Dilation
        filt_dil = sitk.BinaryDilateImageFilter()
        filt_dil.SetKernelRadius(10)

        mask_dil = filt_dil.Execute(mask_s)
        mask_dil_arr = sitk.GetArrayFromImage(mask_dil)
        new_mask_bool = mask_dil_arr.astype(dtype=bool)

        # bbox
        _idx = np.sum(new_mask_bool, axis=(0, 1)) > 0
        new_mask_bool = new_mask_bool[:, :, _idx]
        bbox = extract_bboxes(new_mask_bool)
        #print(bbox)
        x = bbox[0][0]
        y = bbox[0][1]
        width = bbox[0][2]
        height = bbox[0][3]

        # crop image
        image_s = input_image[:,:,i]
        image_arr = sitk.GetArrayFromImage(image_s)
        crop_img = image_arr[x:width, y:height]
        an_array = np.zeros((96,96))
        an_array[:] = np.NaN
        an_array[x:width, y:height] = crop_img
        cropped_img[i,:,:] = an_array

    cropped = sitk.GetImageFromArray(cropped_img)
    cropped.SetSpacing(input_image.GetSpacing())
    cropped.SetOrigin(input_image.GetOrigin())
    new_image = sitk.Cast(cropped, sitk.sitkFloat32)   

    return new_image 

def data_preprocess(im_paths, mask_paths, n_classes, index, new_size, aug, aug_factor, flip, sharp, intense_window, otsu):
    """ Function to carry out pre-processing including data augmentation of images during generation of data
    batches in data generator
    Inputs:
    im_paths - list of image paths
    label_paths - list of label paths
    n_classes - number of classes to be segmented
    index - list of indices from generator to select batch
    new_size - desired size of images
    d_type - whether selecting training or testing data
    aug - whether to carry out augmentation or not
    """
    images = []
    masks = []
    imgs_list = os.listdir(im_paths)
    masks_list = os.listdir(mask_paths)
    for k in index:
        image = sitk.ReadImage(im_paths + imgs_list[k])

        # Pre -processing (rescale)
        image_adjusted = preprocessing_tools.adjust_window_level(image)
        image_eq = preprocessing_tools.specific_intensity_window(image_adjusted, window_percent=intense_window)
            #image_eq = preprocessing_tools.histogram_equalization(sitk.Cast(image_adjusted, sitk.sitkInt32), use_target_range=False) # Cast into float32
            #sitk.WriteImage(image_eq, 'Z:/pruebas/' + str(k) + '.nii')
        image_rescaled = preprocessing_tools.rescale(image_eq, 0, 255) #  + Cast into float32
        
        images.append(sitk.GetArrayFromImage(image_rescaled)[:, :, :, np.newaxis])

        if mask_paths is None:
            masks = None
        else:
            mask = sitk.ReadImage(mask_paths + masks_list[k]) #uint16
            # Set common spacing and origin
            mask.SetSpacing(image_rescaled.GetSpacing())
            mask.SetOrigin(image_rescaled.GetOrigin())
            mask.SetDirection(image_rescaled.GetDirection())
            # sitk.WriteImage(mask, os.path.join('//172.18.93.155/aoyarzun/pruebas/' + str('{:04d}mask.mha'.format(k))))


            # if len(sitk.GetArrayFromImage(mask).shape) > 2:
            #     mask = mask[:, :, 0]
            masks.append(sitk.GetArrayFromImage(mask)[:, :, :, np.newaxis])

        aug_functions = [functools.partial(data_augmentation_tools.rotate, rotation_degrees=[4]),
                        functools.partial(data_augmentation_tools.rotate, rotation_degrees=[6]),
                        functools.partial(data_augmentation_tools.rotate, rotation_degrees=[8]),
                        functools.partial(data_augmentation_tools.rotate, rotation_degrees=[10]),
                        functools.partial(data_augmentation_tools.translate_scipy, translations = [(0, 0, 10)]), 
                        functools.partial(data_augmentation_tools.translate_scipy, translations = [(0, 0, -10)]),
                        functools.partial(data_augmentation_tools.translate_scipy, translations = [(0, 10, 0)]), 
                        functools.partial(data_augmentation_tools.translate_scipy, translations = [(0, -10, 0)]), 
                        functools.partial(data_augmentation_tools.flip_well, axis = [True, False, False]), # en x
                        functools.partial(data_augmentation_tools.flip_well, axis = [False, True, False]) # en y
                         ]

        if aug:
            select_augment = np.arange(len(aug_functions))
            np.random.shuffle(select_augment)
            #select_augment = select_augment[:min(len(aug_functions), aug_factor - 1)]

            for a in select_augment:
                if mask is not None:
                    # if a in np.arange(3):
                    #     new_image = aug_functions[a](image)
                    #     images.append(sitk.GetArrayFromImage(new_image)[:, :, np.newaxis])
                    #     labels.append(sitk.GetArrayFromImage(label)[:,:,np.newaxis])
                    # else:
                    #print(a)           
                    new_image, new_mask = aug_functions[a](image_rescaled, mask)

                    if new_size:
                        new_image_padded = preprocessing_tools.image_padding(new_image, new_size)
                        new_mask_padded = preprocessing_tools.image_padding(new_mask, new_size)

                        new_image = new_image_padded
                        new_mask = new_mask_padded
                        
                    try:
                        images.append(sitk.GetArrayFromImage(new_image[0])[:, :, :, np.newaxis])
                        masks.append(sitk.GetArrayFromImage(new_mask[0])[:, :, :, np.newaxis])
                    except IndexError:
                        images.append(sitk.GetArrayFromImage(new_image)[:, :, :, np.newaxis])
                        masks.append(sitk.GetArrayFromImage(new_mask)[:, :, :, np.newaxis])

                    # print(new_image[0].GetSize())
                    # print(new_image[0].GetSize())
                    # sitk_show(new_image[0][:,:,1])
                    # sitk_show(new_mask[0][:,:,1])
                    #sitk.WriteImage(new_image[0], os.path.join('Z:/pruebas/' + str(a) + '_image.nii'))
                    #sitk.WriteImage(new_mask[0], ['Z:/pruebas/' + str(a) + '_mask.nii'])
                    # if len(new_image[0].GetSize()) == 2:
                    #     new_image_cropped = crop_images(new_image, new_mask)
        
                    #     images.append(sitk.GetArrayFromImage(new_image_cropped)[:, :, :, np.newaxis])
                    #     masks.append(sitk.GetArrayFromImage(new_mask)[:, :, :, np.newaxis])
                    # else:
                    #     new_image_cropped = crop_images(new_image[0], new_mask[0])
                    #     images.append(sitk.GetArrayFromImage(new_image_cropped)[:, :, :, np.newaxis])
                    #     masks.append(sitk.GetArrayFromImage(new_mask[0])[:, :, :, np.newaxis])
                    
                        
            
                    # except ValueError:
                    #     new_image, new_mask, _ = aug_functions[a](image_rescaled, mask)
                        

                    # try:
                    #     images.append(sitk.GetArrayFromImage(new_image_cropped)[:, :, :, np.newaxis])
                    #     masks.append(sitk.GetArrayFromImage(new_mask[0])[:, :, :, np.newaxis])
                    # except IndexError:
                    #     images.append(sitk.GetArrayFromImage(new_image_cropped)[:, :, :, np.newaxis])
                    #     masks.append(sitk.GetArrayFromImage(new_mask)[:, :, :, np.newaxis])
                else:

                    try:
                        new_image, _ = aug_functions[a](image_rescaled, None)
                    except ValueError:
                        new_image, _, _ = aug_functions[a](image_rescaled, None)
                    try:
                        images.append(sitk.GetArrayFromImage(new_image[0])[:, :, :, np.newaxis])
                    except IndexError:
                        images.append(sitk.GetArrayFromImage(new_image)[:, :, :, np.newaxis])

    images = np.array(images)
    
    if masks is not None:
        masks = np.array(masks)
        masks_one_hot = make_one_hot(masks, n_classes)
    else:
        masks_one_hot = None
    
    # Ver que cada imagen tiene la dimensión correcta
    # print(images[0].shape)
    # print(masks_one_hot[0].shape)

    # Ver la longitud de cada array de imagen/mask (original + augmentations)
    # print(len(images))
    # print(len(masks_one_hot))

    # Save to check if images and masks correspond
    # mask_to_check = masks[50][1,:,:]
    #image_to_check = images[0][1,:,:]
    # sitk.WriteImage(sitk.GetImageFromArray(mask_to_check), '//172.18.93.155/aoyarzun/pruebas/mask_to_check.mha')
    #sitk.WriteImage(sitk.GetImageFromArray(image_to_check), '//172.18.93.155/aoyarzun/pruebas/image_to_check.mha')

    return images, masks_one_hot

def make_one_hot(label_set, n_classes):
    """Used in the data_preprocess function
    TODO make more general or make optional, currently specific to class labelling"""
    labels_one_hot = np.ndarray(tuple(label_set.shape[:4]) + (n_classes,), dtype=np.bool) #antes estaba a [:3]

    for i in range(n_classes):
        labels_one_hot[..., i] = label_set[:, :, :, :, 0] == i

    return labels_one_hot


def train(net_hyperparams, unet_builder, log_dir, model_dir, plt_dir, monitor=True, checkpoint=True, plot = False):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #unet_path = os.path.join(model_dir, "unet_{}_{}_{}".format(net_hyperparams['batch_size'], net_hyperparams['epochs'], net_hyperparams['learning_rate']))
    unet_path = model_dir

    unet = KerasNetwork(unet_builder, net_hyperparams)
    opt = net_hyperparams['optimizer']
    if opt == 'adam':
        opt = Adam(net_hyperparams['learning_rate'])

    #unet.model.compile(opt, net_hyperparams['loss'], net_hyperparams['metric'])
    unet.model.compile(optimizer=Adam(lr=net_hyperparams['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=net_hyperparams['loss'], metrics=net_hyperparams['metric'])
    my_callbacks = []

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            #os.makedirs(os.path.join(model_dir + log_dir))
        #log_loc = os.path.join(os.path.join(model_dir + log_dir), 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_loc = os.path.join(log_dir, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        tensor_board = TensorBoard(log_dir=log_loc, histogram_freq=0,
                                    write_graph=True, write_images=False, update_freq=5,
                                    embeddings_freq=0,
                                    embeddings_metadata=None)
        my_callbacks.append(tensor_board)
    if monitor:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8, verbose=0, mode='auto')
        my_callbacks.append(early_stopping)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=3, verbose=0, min_lr=0.00001)
        my_callbacks.append(reduce_lr)
    if checkpoint:
        path = model_dir + '/weights.{epoch:02d}-{val_loss:.2f}.h5'
        save_model = ModelCheckpoint(path, monitor='loss', verbose=0, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=10)
        my_callbacks.append(save_model)
  
    history_unet = unet.model.fit(train_gen,
                                        epochs=net_hyperparams['epochs'], verbose=1,
                                        validation_data=val_gen,                                    
                                     callbacks=my_callbacks,validation_freq=1 )

    unet.model.save(unet_path)

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(history_unet.history['loss'], label="TrainLoss")
        ax[0].plot(history_unet.history['val_loss'], label="ValLoss")
        ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history_unet.history['categorical_accuracy'], label="TrainAcc")
        ax[1].plot(history_unet.history['val_categorical_accuracy'], label="ValAcc")
        ax[1].legend(loc='best', shadow=True)
        try:
            if not os.path.exists(plt_dir):
                os.makedirs(plt_dir)
            plt.savefig(os.path.join(plt_dir, 'UNET_{}_{}_{}.png'.format(net_hyperparams['epochs'], net_hyperparams['batch_size'], net_hyperparams['learning_rate'])))
        except TypeError:
            print('Creating plot folder in current directory')
            os.makedirs('Plots')
            plt.savefig(os.path.join('Plots', 'UNET_{}_{}_{}.png'.format(net_hyperparams['epochs'], net_hyperparams['batch_size'],
                                                                         net_hyperparams['learning_rate'])))


def make_prediction(test_data, saved_model_path, hyperparams, output_pred_path, plot_overlay=False, flip=False):
    """Carry out a prediction with a saved model
    Inputs:
    test_data - generator loaded with test_data
    weights_unet - directory to where UNET weights are stored
    hyperparams - desired hyperparameters
    """

    print('Load model')
    custom_objects = {'GeneralWeightedDiceLoss': GeneralWeightedDiceLoss(weights=[0.1,0.9])}
    #custom_objects = {'dice_coefficient_loss': dice_coefficient_loss}
    unet = load_model(saved_model_path, custom_objects=custom_objects, compile=False)
    tic()
    res_unet = unet.predict(test_data)
    toc()

    label_kidney_unet = res_unet[:, :, :, :, 1]
    #label_fat_unet = res_unet[:, :, :, 2]
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    test_ims = os.listdir(test_data.images_paths)
    #print('shape of res_unet is:', res_unet.shape[0])
    for i in range(res_unet.shape[0]):
        image = sitk.ReadImage(test_data.images_paths + test_ims[i]) # solo se usa para obtener el spacing, size...
        #sitk_show(image[:,:,0])
        #image, flipped = preprocessing_tools.read_and_flip(test_ims[i], None, 1, check_LUT=True, flip=flip)
        size = (image.GetSize()[0], image.GetSize()[1])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        kidney_mask = otsu_filter.Execute(sitk.GetImageFromArray(label_kidney_unet[i, :, :, :]))
        #sitk.WriteImage(kidney_mask, 'C:/Users/anne.oyarzun/Desktop/pruebas/kidney_mask.nii')

        # It is not neccesary to resize, as it has not been resized in the pre-processing step. 
        mask_name = 'kidney'
        #im_mask = sitk.GetImageFromArray(kidney_mask)
        kidney_mask.SetOrigin(origin)
        kidney_mask.SetSpacing(spacing)
        print('Save segmentation as', str('{:04d}'.format(i+2)) + '_%s.mha' % mask_name)
        
        sitk.WriteImage(kidney_mask, os.path.join(output_pred_path + str('{:04d}.mha'.format(i+2))))

def post_process(image):
    Origin=image.GetOrigin()
    Spacing=image.GetSpacing()
    # mask_inverted = postprocessing_tools.invert_mask_intensity(image)

    largest = postprocessing_tools.keep_biggest_object(sitk.GetArrayFromImage(image))

    final_mask = sitk.GetImageFromArray(largest)
    final_mask.SetOrigin(Origin)
    final_mask.SetSpacing(Spacing)

    return final_mask


def get_Dice(gt_segmentation,pred_segmentation, label=1):
    mask_gt=sitk.BinaryThreshold(gt_segmentation,
                        lowerThreshold=label, upperThreshold=label,
                        insideValue=1, outsideValue=0)
    mask_pred=sitk.BinaryThreshold(pred_segmentation,
                        lowerThreshold=label, upperThreshold=label,
                        insideValue=1, outsideValue=0)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(mask_gt,mask_pred)
    Dicce_value=overlap_measures_filter.GetDiceCoefficient()
    return Dicce_value

def load_data(db, im_path, label_path, hyperparams, val_split=0.25, sub_sample_size=None, augment=False,
            aug_factor=8, flip=False, sharpen=False, intense_window=False, otsu=True):
        if db == 'test':
                return DataGenerator(im_path, label_path,
                                    hyperparams['num_classes'], batch_size=1, image_dimensions = [96,96,1],
                                    flip=flip, sharpen=sharpen, intense_window=intense_window, otsu=otsu), None


if __name__== '__main__':

    ##########################################################
    ########### DATA PATH AND HYPERPARAMETERS ################
    ##########################################################
    
    data_path = ''
    
    train_imgs_db = data_path + 'train/Images/'
    train_masks_db = data_path + 'train/Masks/'
    val_imgs_db = data_path + 'val/Images/'
    val_masks_db = data_path + 'val/Masks/'

    print('Training...')

    # MODEL 1. dice + original
    net_hyperparams = {'image_size': ,
                        'learning_rate': ,
                        'epochs': ,
                        'num_classes': ,
                        'batch_size': ,
                        'optimizer': 'adam',
                        'loss': GeneralWeightedDiceLoss(weights=[0.1,0.9]),
                        'metric': ['categorical_accuracy']}


    model_name = "unet_{}_{}_{}".format(net_hyperparams['batch_size'], net_hyperparams['epochs'], net_hyperparams['learning_rate'])
    model_dir = ''
    plt_dir = model_dir
    log_dir = model_dir + 'logs/'

    

    ##########################################################
    ################################ TRAINING ################
    ##########################################################

    train_gen = DataGenerator(train_imgs_db, train_masks_db, n_classes = 2, batch_size = net_hyperparams['batch_size'], image_dimensions= [128,128,3], 
                                shuffle = True, augment = True, aug_factor = 8, flip = True, sharpen=False, 
                                intense_window=0.15, otsu=False)
    val_gen = DataGenerator(val_imgs_db, val_masks_db, n_classes = 2, batch_size = net_hyperparams['batch_size'], image_dimensions= [128,128,3], 
                                shuffle = True, augment = True, aug_factor = 8, flip = True, sharpen=False, 
                                intense_window=0.15, otsu=False)

    unet_builder = OriginalUNETArchitecture3DKerasNetworkBuilder(net_hyperparams)

    tic()
    train(net_hyperparams, unet_builder, log_dir, model_dir, plt_dir, monitor=True, checkpoint=True, plot = True)
    toc()

    ##########################################################
    ################################ TESTING  ################
    ##########################################################
    
    num_classes = 2
    weights = [0.1, 0.9]
    test_imgs_path = ''
    test_labels_path = ''
    db = 'test'

    test_data, _ = load_data(db, test_imgs_path, test_labels_path, net_hyperparams, flip = False, sharpen = False, intense_window=False, otsu=False)
    model_filename = [model_dir + model_name + '/']
    
    result_dir = model_dir + 'results/'
    post_result_dir = model_dir + 'results_postprocessed/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(post_result_dir):
        os.makedirs(post_result_dir)

    test_fullnames = os.listdir(test_data.images_paths)
    test_names = list()


    for test_imgs in range(len(test_fullnames)): 
        filename, _ = os.path.splitext(test_fullnames[test_imgs])
        test_names.append(filename)
    
    print('Testing...')
    tic()
    make_prediction(test_data, model_dir, net_hyperparams, result_dir, plot_overlay=False, flip=False)
    toc()

    ##########################################################
    ##################### EVALUATION #########################
    ##########################################################
    predicted_masks = natsorted(os.listdir(result_dir))
    gt_masks = natsorted(os.listdir('Z:/RM_RENAL/DATASET/test/masks/'))
    dice_scores = []
    metric_table = np.empty(shape = [51, 3])
    evaluate = True

    if evaluate == True:
        for imgs in range(0,len(predicted_masks)): 
            gt_mask = sitk.ReadImage('Z:/RM_RENAL/DATASET/test/masks/' + gt_masks[imgs])
            
            pred_mask = sitk.ReadImage(result_dir + predicted_masks[imgs])
            pred_mask = sitk.Cast(pred_mask, sitk.sitkUInt16)

            pred_mask.SetSpacing(gt_mask.GetSpacing())
            pred_mask.SetOrigin(gt_mask.GetOrigin())
            pred_mask.SetDirection(gt_mask.GetDirection())

            size_pred = pred_mask.GetSize()
            post_mask = []

            postprocessed_pred_mask = []
            
            for size_p in range(0,size_pred[2]):
                pred_mask_slice = pred_mask[:,:,size_p]
                pred_mask_slice = post_process(pred_mask_slice)
                postprocessed_pred_mask.append(sitk.GetArrayFromImage(pred_mask_slice))
                
                post_mask.append(sitk.GetArrayFromImage(pred_mask_slice))
    

                gt_mask_slice = gt_mask[:,:,size_p]
                dice_score = get_Dice(gt_mask_slice, pred_mask_slice)
                metric_table[imgs, size_p] = dice_score

                if dice_score < 0.7:
                    #print('Dice score is less than 0.5, postprocess discarded')
                    pred_mask_slice = pred_mask[:,:,size_p]
                    dice_score = get_Dice(gt_mask_slice, pred_mask_slice)
                    print(dice_score)
                else:
                    print(dice_score)

    
            post_mask = sitk.GetImageFromArray(post_mask)
            sitk.WriteImage(post_mask, os.path.join(post_result_dir + str('{:04d}.mha'.format(imgs+2))))

    np.savetxt(os.path.join(model_dir + 'metrics.csv'), metric_table, delimiter =", ", fmt ='% s')



            









    

    



    



        
