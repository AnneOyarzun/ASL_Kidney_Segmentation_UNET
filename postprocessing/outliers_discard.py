import SimpleITK as sitk
import os
from natsort import os_sorted
from preprocessing.other_functions import tic,toc
from Visualization.itk_visualization import sitk_show
from dataset.extract_slices import create_img_vector
from Registration.create_PWIs import create_mean_PWIs_groupwise
import numpy as np
from preprocessing.preprocessing_tools import rescale
from roipoly import RoiPoly
from matplotlib import pyplot as plt
from preprocessing import gui
import cv2


# Cargar el vector de imágenes registradas
root = 'C:/Users/anne.oyarzun/Documents/REGISTRO/resultados/registro/groupwise/'
transf = 'Transform_BSplineStackTransform/'
params = 'reso_2_iter_1800_interpolation_order_1_3/reg_gw_09_01_metric_pca2_trans_bstack_rebeca_def.nii'
result_dir = root + transf + params
registered_serie = sitk.ReadImage(result_dir)

#sitk.Show(registered_serie)

#imagen = registered_serie[:,:,0]

## Prubas con la nueva herramienta



mask = sitk.ReadImage('C:/Users/anne.oyarzun/Documents/REGISTRO/prueba_mascara_out.mha')
mask = mask[:,:,0]
mask_arr = sitk.GetArrayFromImage(mask)

size_serie = registered_serie.GetSize()
print(size_serie)

# for i in range(int(size_serie[2]/2)):
#     imagen = registered_serie[:,:,i]
#     image = sitk.GetArrayFromImage(imagen)
#
#     locs = np.where(mask_arr == 1)
#     pixels = image[locs]
#     print(np.mean(pixels))
#     print(np.std(pixels))



controls = np.arange(0, size_serie[2], 2).tolist()
labels = np.arange(1, size_serie[2], 2).tolist()

for i in range(len(controls)):

    cont = registered_serie[:,:,controls[i]]
    label = registered_serie[:,:,labels[i]]

    pwi = sitk.Subtract(label,cont)
    image = sitk.GetArrayFromImage(pwi)

    locs = np.where(mask_arr == 1)
    pixels = image[locs]
    print(np.mean(pixels))
    print(np.std(pixels))

