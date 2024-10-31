import SimpleITK as sitk
import numpy as np


def otsu_thresholding(image):
    thresIm = sitk.OtsuThreshold(image)
    return thresIm



def labelBinaryImageAndGetShapeStats( imgBW, backGroundVal=0):
    """
    Execute sitk.ConnectedComponents on binary image and get shapeStats
    :param imgBW: binary image
    :param backGroundVal: background intensity value for shape stats
    :return imgLabel: labelles image
    :return N_Labels: number of labels
    """

    if not isinstance(imgBW, sitk.Image):
        raise (IOError, 'ERROR: the image passed must be in simpleITK Image format')

    imgLabel = sitk.ConnectedComponent(imgBW)
    shapeStats = sitk.LabelShapeStatisticsImageFilter()
    shapeStats.SetBackgroundValue(backGroundVal)
    shapeStats.Execute(imgLabel)
    # N_Labels = shapeStats.GetNumberOfLabels()

    return imgLabel, shapeStats


def keep_biggest_object(image):
    """
    Analize objets in a binary image and keep the biggest one (to
    remove small secondary objects that may appear after segmentation)
    :param image: binary simple itk image
    :return:
    """

    img, shape_stats = labelBinaryImageAndGetShapeStats(sitk.GetImageFromArray(image), backGroundVal=0)
    max_size=0
    label=0
    for i in shape_stats.GetLabels():
        size= shape_stats.GetPhysicalSize(i)
        if size > max_size:
            max_size=size
            label= i
    #print("Label to keep: ", label)

    img = sitk.GetArrayFromImage(img)
    out_image = img

    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            if img[i,j]==label:
                out_image[i, j]=1
            else:
                out_image[i, j]=0

    return out_image

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


