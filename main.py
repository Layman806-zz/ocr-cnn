import h5py
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import numpy as np
import cv2

img_width = 64
img_height = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(62))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.load_weights('first_try.h5')
'''
batch_size = 1
# pr = model.predict_classes(im.reshape((1, 1, img_height, img_width)))
pr_data = ImageDataGenerator(rescale=1./255)
pr_generator = pr_data.flow_from_directory('predict',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode=None)
'''
im = cv2.imread('test.png')
im = cv2.resize(im, (img_height, img_width))
im = np.reshape(im, [1, img_height, img_width, 3])
pr = model.predict(im)

print(pr)
print(np.argmax(pr))
