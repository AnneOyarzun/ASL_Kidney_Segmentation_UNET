import SimpleITK as sitk
import numpy as np


def rescale(image, min=0, max=255):
    """ Rescale image intensity between 0 and 255 (default)
    :param image: image to rescale (simple itk image) :return: rescaled image """
    px_type = image.GetPixelIDTypeAsString()
    #print(px_type)
    if 'int' in px_type:
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        image = castImageFilter.Execute(image)

    out_image_rescaled = sitk.RescaleIntensity(image, min, max)
    return out_image_rescaled

def adjust_window_level( image, windowMin = 0, windowMax = 255):
    imageWL = sitk.IntensityWindowing(image,
                            windowMinimum=windowMin,
                            windowMaximum=windowMax)
    return imageWL


def resize(image, new_size, label=None):
    '''
    Resize image
    :param image: image to resize (simple itk image)
    :param label: label to resize (optional) (usually necessary)
    :param new_size: new image size, as a list (specify size for each dimension eg, for 2d image: [224, 160], for 3d: [224, 160, 20] )
    :return: image resized, label resized (if provided as input)
    '''
    new_size = [int(i) for i in new_size]
    reference_image = sitk.Image(new_size, image.GetPixelIDValue())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    reference_image.SetSpacing(
        [sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())])

    interpolator = sitk.sitkLinear
    identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    interpolator_label = sitk.sitkNearestNeighbor
    out_image_resized = sitk.Resample(image, reference_image, identity, interpolator, 0, sitk.sitkFloat32)
    out_label_resized = None
    if label != None:
        out_label_resized = sitk.Resample(label, reference_image, identity, interpolator_label, 0,

                                          sitk.sitkUInt8)
    # rescale intensities
    return out_image_resized, out_label_resized


def fill_holes(label):
    '''
    It fills the holes (for example, for abdominal aorta, it fills the lumen hole that appears in the mask)
    :param label: label where the holes will be removed (simple itk)
    :return: label with removed holes
    '''

    fillHole=sitk.GrayscaleFillholeImageFilter

    if label.GetDimension()==3:
        x, y, z= label.GetSize()
        np_label = sitk.GetArrayFromImage(label)

        for slice in range(z):

            #para asignar un conjunto de filas y columnas hay que utilizar numpy
            noHole = sitk.GrayscaleFillhole(label[:, :, slice])

            np_label[slice, :, :] = sitk.GetArrayFromImage(noHole)

    elif label.GetDimension==2:
        noHole = sitk.GrayscaleFillhole(label)

    withoutHoles=sitk.GetImageFromArray(np_label)
    withoutHoles.SetOrigin(label.GetOrigin())
    withoutHoles.SetSpacing(label.GetSpacing())

    return withoutHoles


def hist_equalization2(image):
    adap_hist=sitk.AdaptiveHistogramEqualizationImageFilter()
    adap_hist.SetAlpha(0.7)
    adap_hist.SetBeta(0.7)
    image_hist=adap_hist.Execute(image)
    return image_hist


def histogram_equalization(image,
                           min_target_range = None,
                           max_target_range = None,
                           use_target_range = True):
    '''
    Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/70_Data_Augmentation.html
    Histogram equalization of scalar images whose single channel has an integer
    type. The goal is to map the original intensities so that resulting
    histogram is more uniform (increasing the image's entropy).
    Args:
        image (SimpleITK.Image): A SimpleITK scalar image whose pixel type
                                 is an integer (sitkUInt8,sitkInt8...
                                 sitkUInt64, sitkInt64).
        min_target_range (scalar): Minimal value for the target range. If None
                                   then use the minimal value for the scalar pixel
                                   type (e.g. 0 for sitkUInt8).
        max_target_range (scalar): Maximal value for the target range. If None
                                   then use the maximal value for the scalar pixel
                                   type (e.g. 255 for sitkUInt8).
        use_target_range (bool): If true, the resulting image has values in the
                                 target range, otherwise the resulting values
                                 are in [0,1].
    Returns:
        SimpleITK.Image: A scalar image with the same pixel type as the input image
                         or a sitkFloat64 (depending on the use_target_range value).
    '''
    arr = sitk.GetArrayViewFromImage(image)

    i_info = np.iinfo(arr.dtype)
    if min_target_range is None:
        min_target_range = i_info.min
    else:
        min_target_range = np.max([i_info.min, min_target_range])
    if max_target_range is None:
        max_target_range = i_info.max
    else:
        max_target_range = np.min([i_info.max, max_target_range])

    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1
    # using ravel, not flatten, as it does not involve memory copy
    hist = np.bincount((arr-min_val).ravel(), minlength=number_of_bins)
    cdf = np.cumsum(hist)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    res = cdf[arr-min_val]
    if use_target_range:
        res = (min_target_range + res*(max_target_range-min_target_range)).astype(arr.dtype)
    return sitk.GetImageFromArray(res)


def specific_intensity_window(image, window_percent=0.2):
    arr = sitk.GetArrayViewFromImage(image)
    arr = arr.astype('int64')
    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1

    hist = np.bincount((arr - min_val).ravel(), minlength=number_of_bins)
    hist_new = hist[1:]
    total = np.sum(hist_new)
    window_low = window_percent * total
    window_high = (1 - window_percent) * total
    cdf = np.cumsum(hist_new)
    low_intense = np.where(cdf >= window_low) + min_val
    high_intense = np.where(cdf >= window_high) + min_val
    res = sitk.IntensityWindowing(sitk.Cast(image, sitk.sitkFloat32), np.double(low_intense[0][0]),
                                    np.double(high_intense[0][0]), np.double(arr.min()), np.double(arr.max()))
    return res

def image_padding(img, new_shape):
    
    arr = sitk.GetArrayFromImage(img[0][:,:,0])
    # Calcula la cantidad de píxeles de relleno en cada lado del eje x
    fill_x = max(0, new_shape[1] - arr.shape[1]) // 2
    fill_y = max(0, new_shape[0] - arr.shape[0]) // 2

    # Aplica el relleno de ceros utilizando numpy
    padded_img = np.pad(arr, ((fill_y, fill_y), (fill_x, fill_x)), mode='constant', constant_values=0)

    return padded_img