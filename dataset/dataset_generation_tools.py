import os
import random
import SimpleITK as sitk
import numpy as np
from natsort import os_sorted
from preprocessing import preprocessing_tools
from augmentation import data_augmentation_tools
from Visualization.itk_visualization import sitk_show
from sklearn.model_selection import train_test_split
import tensorflow as tf



def Create_Train_Dataset_H5(im_path, mask_path, train_ids, study, modality, aug=False, plot_example=False):
    '''
    Function to generate the Train Dataset using H5DF
    :param im_path: main path to images
    :param mask_path: main path to masks
    :param train_ids: the n-1 ids for training data, avoids mixing patient data.
    :param study: number of study (01 or 02)
    :param modality: modality of images (PCASL, FAIR, IVIM, T1)
    :param plot_example: set True to check the array has been created correctly
    :return: array of images and masks and corresponding metadadta
    '''

    all_imgs_train_arr = []
    all_masks_train_arr = []
    all_imgs_metadata_for_h5 = []
    all_masks_metadata_for_h5 = []

    patient_list = os.listdir(im_path)

    for pats_train in range(len(train_ids)):
        imgs_to_train = os_sorted(
            os.listdir(im_path + patient_list[train_ids[pats_train]] + '/' + study + '/' + modality + '/'))
        labels_to_train = os_sorted(
            os.listdir(mask_path + patient_list[train_ids[pats_train]] + '/' + study + '/' + modality + '/'))

        for file in range(len(imgs_to_train)):
            image = sitk.ReadImage(
                im_path + patient_list[train_ids[pats_train]] + '/' + study + '/' + modality + '/' + imgs_to_train[
                    file])
            mask = sitk.ReadImage(
                mask_path + patient_list[train_ids[pats_train]] + '/' + study + '/' + modality + '/' + labels_to_train[
                    file])

            # print('imagen')
            # print(image.GetOrigin())
            # print(image.GetSpacing())
            # print(image.GetDirection())
            # print(image.GetSize())

            # print('mask')
            # print(mask.GetOrigin())
            # print(mask.GetSpacing())
            # print(mask.GetDirection())
            # print(mask.GetSize())

            mask.SetOrigin(image.GetOrigin())
            mask.SetSpacing(image.GetSpacing())

            # print(mask.GetOrigin())
            # print(mask.GetSpacing())
            # print(mask.GetDirection())
            # print(mask.GetSize())

            image_metadata = {}
            mask_metadata = {}
            image_metadata['origin'] = image.GetOrigin()
            image_metadata['spacing'] = image.GetSpacing()
            image_metadata['type'] = image.GetPixelIDTypeAsString()
            image_metadata['original size'] = image.GetSize()

            mask_metadata['origin'] = mask.GetOrigin()
            mask_metadata['spacing'] = mask.GetSpacing()
            mask_metadata['type'] = mask.GetPixelIDTypeAsString()
            mask_metadata['original size'] = mask.GetSize()

            im_rescaled = preprocessing_tools.adjust_window_level(image)
            im_rescaled = preprocessing_tools.rescale(im_rescaled, 0, 255)

            if aug:
                rotated_images, rotated_masks = data_augmentation_tools.rotate(im_rescaled, mask,
                                                                               rotation_degrees=[2, 4, 6, 8, 10,
                                                                                                 12, 14, 16])
                translated_images, translated_masks = data_augmentation_tools.translate_scipy(im_rescaled,
                                                                                              mask,
                                                                                              translations=[
                                                                                                  (5, 0, 0),
                                                                                                  (0, 5, 0),
                                                                                                  (0, 0, 5),
                                                                                                  (5, 5, 0),
                                                                                                  (5, 0, 5),
                                                                                                  (0, 5, 5),
                                                                                                  (5, 5, 5),
                                                                                                  (9, 0, 0),
                                                                                                  (0, 9, 0),
                                                                                                  (0, 0, 9)])
                flipped_images, flipped_masks = data_augmentation_tools.flip(im_rescaled, mask)

                for r in range(len(rotated_images)):
                    image_metadata = {}
                    mask_metadata = {}

                    image_metadata['origin'] = rotated_images[r].GetOrigin()
                    image_metadata['spacing'] = rotated_images[r].GetSpacing()
                    image_metadata['type'] = rotated_images[r].GetPixelIDTypeAsString()
                    image_metadata['original size'] = image.GetSize()

                    mask_metadata['origin'] = rotated_masks[r].GetOrigin()
                    mask_metadata['spacing'] = rotated_masks[r].GetSpacing()
                    mask_metadata['type'] = rotated_masks[r].GetPixelIDTypeAsString()
                    mask_metadata['original size'] = mask.GetSize()

                    all_imgs_train_arr.append(sitk.GetArrayFromImage(rotated_images[r]))
                    all_masks_train_arr.append(sitk.GetArrayFromImage(rotated_masks[r]))
                    all_imgs_metadata_for_h5.append(image_metadata)
                    all_masks_metadata_for_h5.append(mask_metadata)
                    print('rotated')

                    if r == 1:
                        imagen_comprobar = rotated_images[r]
                        # sitk.WriteImage(imagen_comprobar, '/data/images/im_comprobar.mha')

                        label_comprobar = rotated_masks[r]
                        # sitk.WriteImage(label_comprobar, '/data/images/label_comprobar.mha')

                for it in range(len(translated_images)):
                    image_metadata = {}
                    label_metadata = {}
                    image_metadata['origin'] = translated_images[it].GetOrigin()
                    image_metadata['spacing'] = translated_images[it].GetSpacing()
                    image_metadata['type'] = translated_images[it].GetPixelIDTypeAsString()
                    image_metadata['original size'] = image.GetSize()

                    label_metadata['origin'] = translated_masks[it].GetOrigin()
                    label_metadata['spacing'] = translated_masks[it].GetSpacing()
                    label_metadata['type'] = translated_masks[it].GetPixelIDTypeAsString()
                    label_metadata['original size'] = mask.GetSize()

                    all_imgs_train_arr.append(sitk.GetArrayFromImage(translated_images[it]))
                    all_masks_train_arr.append(sitk.GetArrayFromImage(translated_masks[it]))
                    all_imgs_metadata_for_h5.append(image_metadata)
                    all_masks_metadata_for_h5.append(mask_metadata)
                    print('translated')

                for f in range(len(flipped_images)):
                    image_metadata = {}
                    label_metadata = {}
                    image_metadata['origin'] = translated_images[f].GetOrigin()
                    image_metadata['spacing'] = translated_images[f].GetSpacing()
                    image_metadata['type'] = translated_images[f].GetPixelIDTypeAsString()
                    image_metadata['original size'] = image.GetSize()

                    label_metadata['origin'] = translated_masks[f].GetOrigin()
                    label_metadata['spacing'] = translated_masks[f].GetSpacing()
                    label_metadata['type'] = translated_masks[f].GetPixelIDTypeAsString()
                    label_metadata['original size'] = mask.GetSize()

                    all_imgs_train_arr.append(sitk.GetArrayFromImage(translated_images[f]))
                    all_masks_train_arr.append(sitk.GetArrayFromImage(translated_masks[f]))
                    all_imgs_metadata_for_h5.append(image_metadata)
                    all_masks_metadata_for_h5.append(mask_metadata)
                    print('translated')

                del rotated_images
                del rotated_masks
                del translated_images
                del translated_masks
                del flipped_images
                del flipped_masks

        else:
            all_imgs_train_arr.append(sitk.GetArrayFromImage(im_rescaled))
            all_masks_train_arr.append(sitk.GetArrayFromImage(mask))

            all_imgs_metadata_for_h5.append(image_metadata)
            all_masks_metadata_for_h5.append(mask_metadata)

    # Check size of the list → [102, 3, 96, 96]
    if plot_example:
        print(np.shape(all_imgs_train_arr))
        all_imgs_train = sitk.GetImageFromArray(all_imgs_train_arr)
        example_img = sitk.GetImageFromArray(all_imgs_train_arr[:][:][:][0])  # cogemos la primera imagen
        example_mask = sitk.GetImageFromArray(all_masks_train_arr[:][:][:][0])  # cogemos la primera máscara
        sitk_show(example_img[:, :, 1])
        sitk_show(example_mask[:, :, 1])

    return all_imgs_train_arr, all_masks_train_arr

def Create_Test_Dataset_H5(im_path, test_ids, study, modality, plot_example=False):
    all_imgs_test_arr = []
    all_imgs_metadata_for_h5 = []

    patient_list = os.listdir(im_path)

    imgs_to_test = os_sorted(os.listdir(im_path + patient_list[test_ids] + '/' + study + '/' + modality + '/'))

    for file in range(len(imgs_to_test)):
        image = sitk.ReadImage(
            im_path + patient_list[test_ids] + '/' + study + '/' + modality + '/' + imgs_to_test[file])
        # print('imagen')
        # print(image.GetOrigin())
        # print(image.GetSpacing())
        # print(image.GetDirection())
        # print(image.GetSize())

        image_metadata = {}
        image_metadata['origin'] = image.GetOrigin()
        image_metadata['spacing'] = image.GetSpacing()
        image_metadata['type'] = image.GetPixelIDTypeAsString()
        image_metadata['original size'] = image.GetSize()

        im_rescaled = preprocessing_tools.adjust_window_level(image)
        im_rescaled = preprocessing_tools.rescale(im_rescaled, 0, 255)

        all_imgs_test_arr.append(sitk.GetArrayFromImage(im_rescaled))
        all_imgs_metadata_for_h5.append(image_metadata)

    return all_imgs_test_arr


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder)
    m = os.listdir(mask_folder)
    random.shuffle(n)
    img = []
    mask = []

    while(True):
        

        for i in range(c,c+batch_size):
            train_img = sitk.ReadImage(img_folder + n[i])
            train_img = preprocessing_tools.adjust_window_level(train_img)
            train_img = preprocessing_tools.rescale(train_img, 0, 255)
            train_mask = sitk.ReadImage(mask_folder + m[i])

            # img[...,np.newaxis] = train_img
            # mask[...,np.newaxis] = train_mask

            img.append(sitk.GetArrayFromImage(train_img))
            mask.append(sitk.GetArrayFromImage(train_mask))


            c+= batch_size
            if(c+batch_size>=len(os.listdir(img_folder))):
                c = 0
                random.shuffle(n)
                random.shuffle(m)
                print('Randomizing again')
            
            print(np.shape(img))

            
            yield img, mask

