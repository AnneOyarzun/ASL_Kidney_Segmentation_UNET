'''
Useful functions for data augmentation (rotations, translations, flip images...)
'''

import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import rotate as scipyrotate
from scipy.ndimage import shift
from numpy import expand_dims
import cv2


def translate_scipy(image, label=None, translations=None, is_color=False):
    """
    Apply small translations to an image (and label if passed as an input). Returns 3 images if input dimension is 2 (3 different

    translations, (5, 0), (0, 5), (5, 5), 7 images if input dimension is 3 ( translations: (5,0,0),(0,5,0),(0,0,5),(5,5,0),(5,0,5),(0,5,5),(5,5,5)
    by default, or the number of images according to the translations provided as input.
    :param image: image
    :param label: label (optional) (usuallye necessary)
    :param dimension: dimension, 2 by default
    :return: translated_images, translated_labels - list containing the translated images
    """
    dimension = image.GetDimension()

    if translations == None:
        translations = [(-5, 0), (0, -5), (-5, -5)]
        if dimension == 3:
            translations = [(-5, 0, 0), (0, -5, 0), (0, 0, -5), (-5, -5, 0), (-5, 0, -5), (0, -5, -5), (-5, -5, -5)]
        elif dimension != 2:
            print("Error translating images. Dimension is not valid. Please set it to 2 or 3 (supported values)")
    else:

        translations = translations

    translated_images = []
    translated_labels = []

    image_array = sitk.GetArrayFromImage(image)

    for trans in translations:
        if is_color:
            translated_image = np.zeros(image_array.shape)
            for i in range(image_array.shape[-1]):
                translated_channel = shift(image_array[:, :, :, i], trans)
                translated_image[:, :, :, i] = translated_channel
        else:
            translated_image = shift(image_array, trans)

        translated_image_itk = sitk.GetImageFromArray(translated_image)
        translated_image_itk.SetOrigin(image.GetOrigin())
        translated_image_itk.SetSpacing(image.GetSpacing())
        translated_images.append(translated_image_itk)

        if label != None:
            label_array = sitk.GetArrayFromImage(label)
            translated_label = shift(label_array, trans)
            translated_label_itk = sitk.GetImageFromArray(translated_label)
            translated_label_itk.SetOrigin(image.GetOrigin())
            translated_label_itk.SetSpacing(image.GetSpacing())
            translated_labels.append(translated_label_itk)

    return translated_images, translated_labels


def translate(image, label=None, translations=None):
    """
    Apply small translations to an image (and label if passed as an input). Returns 3 images if input dimension is 2 (3 different

    translations, (5, 0), (0, 5), (5, 5), 7 images if input dimension is 3 ( translations: (5,0,0),(0,5,0),(0,0,5),(5,5,0),(5,0,5),(0,5,5),(5,5,5)
    by default, or the number of images according to the translations provided as input.
    :param image: image
    :param label: label (optional) (usuallye necessary)
    :param dimension: dimension, 2 by default
    :return: translated_images, translated_labels - list containing the translated images
    """
    dimension = image.GetDimension()
    reference_image = image
    interpolator_for_image = sitk.sitkCosineWindowedSinc
    interpolator_for_label = sitk.sitkNearestNeighbor
    default_value_image = 0  # air hounsfield unit
    default_value_label = 0

    if translations == None:
        translations = [(5, 0), (0, 5), (5, 5)]
        if dimension == 3:
            translations = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0), (5, 0, 5), (0, 5, 5), (5, 5, 5)]
        elif dimension != 2:
            print("Error translating images. Dimension is not valid. Please set it to 2 or 3 (supported values)")
    else:
        translations = translations

    translated_images = []
    translated_labels = []

    for trans in translations:
        # x_trans, y_trans, z_trans = trans
        # [x_trans, y_trans] = trans
        affine_transform = sitk.AffineTransform(dimension)
        if dimension == 2:
            x_trans, y_trans = trans
            affine_transform.SetTranslation((x_trans, y_trans))
        if dimension == 3:
            x_trans, y_trans, z_trans = trans
            affine_transform.SetTranslation((x_trans, y_trans, z_trans))

        translated_image = sitk.Resample(image, reference_image, affine_transform, interpolator_for_image,
                                         default_value_image)
        # print('size translated image = ' + str(translated_image.GetSize()))
        translated_images.append(translated_image)
        if label != None:
            translated_label = sitk.Resample(label, reference_image, affine_transform, interpolator_for_label,
                                             default_value_label)
            translated_labels.append(translated_label)

    return translated_images, translated_labels


# def flip(image, label=None, axis = [None, None, None]): # change if the axis are 3

#     flipped_image = sitk.Flip(image, axis)

#     if label != None:
#         flipped_label = sitk.Flip(label, axis)

#     return flipped_image, flipped_label

def flip_well(image_to_flip, mask_to_flip, axis = [False, False, False]): 
    flipped_img = sitk.Flip(image_to_flip, axis)
    flipped_mask = sitk.Flip(mask_to_flip, axis)

    return flipped_img, flipped_mask

def flip(image, label=None, axis=[True, False, False]):
    '''
    Flips image over specified axes
    :param image: original image to flip
    :param label: original label to flip
    :param axis: axis over which the image will be flipped, default y axis --> [True, False, False]
                'True' is the value that has to be introduced in order to choose the axis:
                x: [False, True, False]
                z: [False, False, True]
    :return: flipped image (numpy array)
    '''

    flip_filter=sitk.FlipImageFilter()

    flip_filter.SetFlipAxes(FlipAxes=axis)
    flip_filter.FlipAboutOriginOn()

    #Execute flips the image across specified axes
    flipped_image=flip_filter.Execute(image)

    flipped_label=[]

    if label!=None:
        flipped_label=flip_filter.Execute(label)

    return flipped_image, flipped_label


def rotate(image, label=None, rotation_degrees=[2, 4, 6, 8, 10]):
    """
    Apply small rotations to input image. Returns 5 images, with different rotations: [2, 4, 6, 8, 10] (degrees) (by default),
    or a number of images according to the provided rotations degrees as input

    :param image - simple itk image

    :param label - values different from zero will be treated as a label
    :param dimension
    :return: two lists, one with the rotated images and other with the rotated labels
    """
    dimension = image.GetDimension()
    rotated_images = []
    rotated_labels = []

    for rot_degree in rotation_degrees:

        if dimension == 2:

            image_array = sitk.GetArrayFromImage(image)
            #label_array = sitk.GetArrayFromImage(label)

            rotated_image = scipyrotate(image_array, rot_degree, axes=(1, 0), order=0, reshape=False)
            rotated_image_itk = sitk.GetImageFromArray(rotated_image)
            rotated_image_itk.SetOrigin(image.GetOrigin())
            rotated_image_itk.SetSpacing(image.GetSpacing())
            rotated_images.append(rotated_image_itk)
            if label != None:
                label_array = sitk.GetArrayFromImage(label)
                rotated_label = scipyrotate(label_array, rot_degree, axes=(1, 0), order=0, reshape=False)
                rotated_label_itk = sitk.GetImageFromArray(rotated_label)
                rotated_label_itk.SetOrigin(label.GetOrigin())
                rotated_label_itk.SetSpacing(label.GetSpacing())
                rotated_labels.append(rotated_label_itk)
        else:

            image_array = sitk.GetArrayFromImage(image)

            rotated_image = scipyrotate(image_array, rot_degree, axes=(1, 2), order=0, reshape=False)
            rotated_image_itk = sitk.GetImageFromArray(rotated_image)
            rotated_image_itk.SetOrigin(image.GetOrigin())
            rotated_image_itk.SetSpacing(image.GetSpacing())
            # print('size rotated image' + str(rotated_image_itk.GetSize()))
            rotated_images.append(rotated_image_itk)
            if label != None:
                label_array = sitk.GetArrayFromImage(label)
                rotated_label = scipyrotate(label_array, rot_degree, axes=(1, 2), order=0, reshape=False)
                rotated_label_itk = sitk.GetImageFromArray(rotated_label)
                rotated_label_itk.SetOrigin(label.GetOrigin())
                rotated_label_itk.SetSpacing(label.GetSpacing())
                # print('size rotated label' + str(rotated_image_itk.GetSize()))

                rotated_labels.append(rotated_label_itk)

    return rotated_images, rotated_labels





def zoom(image, label = None): 

    zoomed_img = expand_dims(sitk.GetArrayFromImage(image), axis = 1)

    if label != None:
        zoomed_label = expand_dims(sitk.GetArrayFromImage(label), axis = 1)



    return sitk.GetImageFromArray(zoomed_img), sitk.GetImageFromArray(zoomed_label)



def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)