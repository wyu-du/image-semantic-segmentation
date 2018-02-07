# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:25:15 2018

@author: think
"""

import os
import sys
import random 
import warnings
import numpy as np
import pandas as pd

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras import models
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# set params
IMG_WIDTH=128
IMG_HEIGHT=128
IMG_CHANNELS=3
TRAIN_PATH='../input/stage1_train/'
TEST_PATH='../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed=42
random.seed=seed
np.random.seed=seed

# get train and test IDs
train_ids=next(os.walk(TRAIN_PATH))[1]
test_ids=next(os.walk(TEST_PATH))[1]

# get and resize train images and masks
X_train=np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train=np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ...')
sys.stdout.flush()
for n, id_ in enumerate(train_ids):
    path=TRAIN_PATH+id_
    img=imread(path+'/images/'+id_+'.png')[:,:,:IMG_CHANNELS]
    img=resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n]=img
    mask=np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path+'/masks/'))[2]:
        mask_=imread(path+'/masks/'+mask_file)
        mask_=np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask=np.maximum(mask,mask_)
    Y_train[n]=mask
    
# get and resize test images
X_test=np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test=[]
print('Getting and resizing test images ...')
sys.stdout.flush()
for n, id_ in enumerate(test_ids):
    path=TEST_PATH+id_
    img=imread(path+'/images/'+id_+'.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img=resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n]=img   
print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, num_classes=2)
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score/t)
    
    K.get_session().run(tf.local_variables_initializer())
    for t in np.arange(0.5, 1.0, 0.05):
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# crop o1 with regard to o2
def crop(o1, o2, img_input):
    o_shape1=Model(img_input, o1).output_shape
    outputHeight1=o_shape1[1]
    outputWidth1=o_shape1[2]
    
    o_shape2=Model(img_input, o2).output_shape
    outputHeight2=o_shape2[1]
    outputWidth2=o_shape2[2]
    
    cx=abs(outputWidth1-outputWidth2)
    cy=abs(outputHeight1-outputHeight2)
    
    # 对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪
    if outputWidth1>outputWidth2:
        o1=layers.Cropping2D(cropping=((0,0),(0,cx)), data_format='channels_last')(o1)
    else:
        o2=layers.Cropping2D(cropping=((0,0),(0,cx)), data_format='channels_last')(o2)
        
    if outputHeight1>outputHeight2:
        o1=layers.Cropping2D(cropping=((0,cy),(0,0)), data_format='channels_last')(o1)
    else:
        o2=layers.Cropping2D(cropping=((0,cy),(0,0)), data_format='channels_last')(o2)
        
    return o1,o2

# Build FCN8 model
n_classes=1
inputs=Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s=Lambda(lambda x: x/255.)(inputs)
    
c1=Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(s)
c1=Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last')(c1)
p1=MaxPooling2D((2,2), strides=(2,2), name='block1_pool', data_format='channels_last')(c1)
f1=p1
    
c2=Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last')(p1)
c2=Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last')(c2)
p2=MaxPooling2D((2,2), strides=(2,2), name='block2_pool', data_format='channels_last')(c2)
f2=p2
    
c3=Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last')(p2)
c3=Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last')(c3)
c3=Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last')(c3)
p3=MaxPooling2D((2,2), strides=(2,2), name='block3_pool', data_format='channels_last')(c3)
f3=p3
    
c4=Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last')(p3)
c4=Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last')(c4)
c4=Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last')(c4)
p4=MaxPooling2D((2,2), strides=(2,2), name='block4_pool', data_format='channels_last')(c4)
f4=p4
    
c5=Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last')(p4)
c5=Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last')(c5)
c5=Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last')(c5)
p5=MaxPooling2D((2,2), strides=(2,2), name='block5_pool', data_format='channels_last')(c5)
f5=p5
    
o=f5
o=(Conv2D(4096, (7,7), activation='relu', padding='same', data_format='channels_last'))(o)
o=Dropout(0.5)(o)
o=(Conv2D(4096, (1,1), activation='relu', padding='same', data_format='channels_last'))(o)
o=Dropout(0.5)(o)
o=(Conv2D(n_classes, (1,1), kernel_initializer='he_normal', data_format='channels_last'))(o)
o=(Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, data_format='channels_last'))(o)
    
o2=f4
o2=(Conv2D(n_classes, (1,1), kernel_initializer='he_normal', data_format='channels_last'))(o2)
o,o2=crop(o,o2,inputs)
o=layers.Add()([o,o2])
o=(Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, data_format='channels_last'))(o)
    
o2=f3
o2=(Conv2D(n_classes, (1,1), kernel_initializer='he_normal', data_format='channels_last'))(o2)
o2,o=crop(o2,o,inputs)
o=layers.Add()([o2,o])
o=(Conv2DTranspose(n_classes, kernel_size=(16,16), strides=(8,8), use_bias=False, padding='same', data_format='channels_last'))(o)

model=Model(inputs,o)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

# fit model
earlystopper=EarlyStopping(patience=5, verbose=1)
#checkpointer=ModelCheckpoint('model-20180206-fcn8.h5', verbose=1, save_best_only=True)
results=model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=10, callbacks=[earlystopper])

# predict on train, val and test
model=load_model('model-20180206-fcn8.h5', custom_objects={'mean_iou': mean_iou})
preds_train=model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val=model.predict(X_train[:int(X_train.shape[0]*0.1)], verbose=1)
preds_test=model.predict(X_test, verbose=1)


# threshold predictions
preds_train_t=(preds_train>0.5).astype(np.uint8)
preds_val_t=(preds_val>0.5).astype(np.uint8)
preds_test_t=(preds_test>0.5).astype(np.uint8)

# create list of upsampled test masks
preds_test_upsampled=[]
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))
    

# run-length encoding
def rle_encoding(x):
    dots=np.where(x.T.flatten()==1)[0]
    run_lengths=[]
    prev=-2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1]+=1
        prev=b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img=label(x>cutoff)
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)
        
new_test_ids=[]
rles=[]
for n, id_ in enumerate(test_ids):
    rle=list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_]*len(rle))
    
# create submission dataframe
sub=pd.DataFrame()
sub['ImageId']=new_test_ids
sub['EncodedPixels']=pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission-20180206-fcn8.csv', index=False)