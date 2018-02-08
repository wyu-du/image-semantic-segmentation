# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:45:04 2018

@author: Wanyu DU
"""

import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt

training_paths=pathlib.Path('input/stage1_train').glob('*/images/*.png')
training_sorted=sorted([x for x in training_paths])
im_path=training_sorted[45]
im=imageio.imread(str(im_path))

# deal with color
print('Original image shape: {}'.format(im.shape))
# coerce the image into grayscale foramt 
from skimage.color import rgb2gray
im_gray=rgb2gray(im)
print('New image shape: {}'.format(im_gray.shape))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(im)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(im_gray,cmap='gray')
plt.title('Grayscale Image')
plt.tight_layout()
plt.show()

# remove background
from skimage.filters import threshold_otsu
thresh_val=threshold_otsu(im_gray)
mask=np.where(im_gray>thresh_val, 1, 0)
# make sure the larger portion of the mask is considered background
if np.sum(mask==0)<np.sum(mask==1):
    mask=np.where(mask,0,1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
im_pixels=im_gray.flatten()
plt.hist(im_pixels,bins=50)
plt.vlines(thresh_val,0,100000,linestyle='--')
plt.ylim([0,50000])
plt.title('Grayscale Histogram')
plt.subplot(1,2,2)
mask_for_display=np.where(mask,mask,np.nan)
plt.imshow(im_gray,cmap='gray')
plt.imshow(mask_for_display,cmap='rainbow',alpha=0.5)
plt.axis('off')
plt.title('Image w/Mask')
plt.show()

# derive individual masks for each object
from scipy import ndimage
labels, nlabels=ndimage.label(mask)
label_arrays=[]
for label_num in range(1, nlabels+1):
    label_mask=np.where(labels==label_num, 1, 0)
    label_arrays.append(label_mask)
print('There are {} separate components/objects detected.'.format(nlabels))
# create a random color map
from matplotlib.colors import ListedColormap
rand_cmap=ListedColormap(np.random.rand(256,3))
labels_for_display=np.where(labels>0,labels,np.nan)
plt.imshow(im_gray,cmap='gray')
plt.imshow(labels_for_display,cmap=rand_cmap)
plt.axis('off')
plt.title('Labeled Cells ({} Nuclei)'.format(nlabels))
plt.show()

# iterate through our masks
for label_ind,label_coords in enumerate(ndimage.find_objects(labels)):
    cell=im_gray[label_coords]
    # check if the label size is too small
    if np.product(cell.shape)<10:
        print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask=np.where(labels==label_ind+1,0,mask)
#regenerate the labels
labels,nlabels=ndimage.label(mask)
print('There are now {} separate components/objects detected.'.format(nlabels))
fig,axes=plt.subplots(1,6,figsize=(10,6))
for ii,obj_indices in enumerate(ndimage.find_objects(labels)[0:6]):
    cell=im_gray[obj_indices]
    axes[ii].imshow(cell,cmap='gray')
    axes[ii].axis('off')
    axes[ii].set_title('Label #{}\nSize: {}'.format(ii+1,cell.shape))
plt.tight_layout()
plt.show()

# open up the differences between two cells
# get the object indices, and perform a binary opening procedure
two_cell_indices=ndimage.find_objects(labels)[1]
cell_mask=mask[two_cell_indices]
cell_mask_opened=ndimage.binary_opening(cell_mask,iterations=8)
fig,axes=plt.subplots(1,4,figsize=(12,4))
axes[0].imshow(im_gray[two_cell_indices],cmap='gray')
axes[0].set_title('Original object')
axes[1].imshow(mask[two_cell_indices],cmap='gray')
axes[1].set_title('Original mask')
axes[2].imshow(cell_mask_opened,cmap='gray')
axes[2].set_title('Opened mask')
axes[3].imshow(im_gray[two_cell_indices]*cell_mask_opened,cmap='gray')
axes[3].set_title('Opened object')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# convert each labeled object to Run Line Encoding
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])
print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))

# combine it into a function
import pandas as pd
def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # read in data and convert to grayscale
    im_id=im_path.parts[-3]
    im=imageio.imread(str(im_path))
    im_gray=rgb2gray(im)
    
    # mask out background and extract connected objects
    thresh_val=threshold_otsu(im_gray)
    mask=np.where(im_gray>thresh_val,1,0)
    if np.sum(mask==0)<np.sum(mask==1):
        mask=np.where(mask,0,1)
        labels,nlabels=ndimage.label(mask)
    labels,nlabels=ndimage.label(mask)
    
    # loop through labels and add each to a DataFrame
    im_df=pd.DataFrame()
    for label_num in range(1,nlabels+1):
        label_mask=np.where(labels==label_num,1,0)
        if label_mask.flatten().sum()>10:
            rle=rle_encoding(label_mask)
            s=pd.Series({'ImageId':im_id,'EncodedPixels':rle})
            im_df=im_df.append(s,ignore_index=True)
    
    return im_df

testing=pathlib.Path('input/stage1_train').glob('*/images/*.png')
for im_path in testing:
    df=analyze_image(im_path)
    df.to_csv('submission.csv',index=None)  
    break