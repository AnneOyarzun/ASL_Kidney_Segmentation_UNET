import SimpleITK as sitk

import os
from natsort import os_sorted


directory = 'C:/Users/anne.oyarzun/Documents/REGISTRO/Dataset_by_slice_TRACTRL09/Slice_01/'
output_dir = 'C:/Users/anne.oyarzun/Documents/REGISTRO/Dataset_by_slice_TRACTRL09/mhd/Slice_01/'
lista = os.listdir(directory)
lista = os_sorted(lista)
print(lista)

for i in range(len(lista)):
    img = sitk.ReadImage(directory + lista[i])
    sitk.WriteImage(img,output_dir + str(i) +'.mha')
