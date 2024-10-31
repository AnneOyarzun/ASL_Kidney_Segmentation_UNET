import SimpleITK as sitk
import os
from natsort import os_sorted


from augmentation.data_augmentation_tools import flip
from preprocessing.preprocessing_tools import fill_holes


def create_img_vector(directory,ref = None, type = None, slice = None):

    lista = os_sorted(os.listdir(directory))


    if ref == True:
        indxs = lista[0::1]
        #print(indxs)

    else:
        indxs = lista[1::1]
        #print(indxs)

    lista_path = []


    for i in range(len(indxs)):
        #img_rot,_ = data_augmentation_tools.rotate(img, label=None, rotation_degrees=[180])
        #img_flip,_ = data_augmentation_tools.flip(img_rot, label=None, axis=[True, False, False])

        lista_path.append(os.path.join(directory, indxs[i]))
        #print(len(lista_path))

        vectorOfImages = sitk.VectorOfImage()



    for filename in range(len(lista_path)):
        if type == 'images': 
            img = sitk.ReadImage(lista_path[filename], sitk.sitkInt16) # Important! cast to sitk.sitkInt16 if images and to sitk.sitkUint16 for masks
        if type == 'masks':
            img = sitk.ReadImage(lista_path[filename], sitk.sitkUInt8) # Important! cast to sitk.sitkInt16 if images and to sitk.sitkUint8 for masks
            img = fill_holes(img)
        img = img[:,:,slice-1]

    

        vectorOfImages.push_back(img)

    image = sitk.JoinSeries(vectorOfImages)
    #sitk.WriteImage(image,'C:/Users/anne.oyarzun/Documents/REGISTRO/serie_imagenes.nii')
    #print(image.GetSize())


    return image


def create_img_vector_asl_t1(directory, slice = None, ref = True, type = None):

    '''
    Creates a vector of images to register. 
    directory: a list of directories, for each image modality (in this example 2; asl and t1)
    ref: if we want to consider the first image of the study. Specific for ASL where there is a M0 image.
    type: if there are images or masks
    '''
        
    lista1 = os_sorted(os.listdir(directory[0]))
    idxasl = len(lista1)
    lista2 = os_sorted(os.listdir(directory[1]))
    idxt1 = len(lista2)
    lista = lista1 + lista2

    lista_path = []


    for i in range(len(lista)):
  
        if i >= idxasl: # All ASL images have been passed
            lista_path.append(os.path.join(directory[1], lista[i]))
        else:
            lista_path.append(os.path.join(directory[0], lista[i]))

        vectorOfImages = sitk.VectorOfImage()

    # Set Common Origin, Spacing and Direction (i.e. first image of asl)
    img_asl = sitk.ReadImage(lista_path[0])
    img_asl = img_asl[:,:,slice-1]
    origin_asl = img_asl.GetOrigin()
    spacing_asl = img_asl.GetSpacing()
    direction_asl = img_asl.GetDirection()

    for filename in range(len(lista_path)):
        if type == 'images': 
            img = sitk.ReadImage(lista_path[filename], sitk.sitkInt16) # Important! cast to sitk.sitkInt16 if images and to sitk.sitkUint16 for masks
        if type == 'masks':
            img = sitk.ReadImage(lista_path[filename], sitk.sitkUInt8) # Important! cast to sitk.sitkInt16 if images and to sitk.sitkUint8 for masks
            img = fill_holes(img)

        img = img[:,:,slice]
        #img = sitk.Extract(img, (img.GetWidth(), img.GetHeight(), 0), (0, 0, 0)) # Very important. It takes images as 2D. 

        img.SetOrigin(origin_asl) # Set commong origin, direction and spacing to all images
        img.SetSpacing(spacing_asl)
        img.SetDirection(direction_asl)

        if filename >= len(lista1):
            img, _ = flip(img)
            img.SetOrigin(origin_asl)
            img.SetSpacing(spacing_asl)
            img.SetDirection(direction_asl)


   
        
        vectorOfImages.append(img)
        

        image = sitk.JoinSeries(vectorOfImages)
        #print(image.GetSize())


    return image
