from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import h5py
import jsonio as jio

datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.2,
        zoom_range=0.0,
        horizontal_flip=False,
        fill_mode='nearest')

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

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(36))  # 36 classes. 0-9 and A-Z
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 16

# augmentation configuration for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.0,
        horizontal_flip=True)

# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# generator reads from directory and assignes classes automatically
train_generator = train_datagen.flow_from_directory(
        'Train',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 64x64
        batch_size=batch_size,
        class_mode='categorical')

# similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'Test',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical')
'''
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('third_try.h5')  # saving weights after training
'''
prev = ''
class_labels = []  # generate class_labels from filenames in test generator's data_flow
for x in validation_generator.filenames:
    i = 0
    while x[i] != '/':
        i = i + 1
    s = x[:i]
    if prev != s:
        class_labels.append(s)
        prev = s

jio.put(class_labels, 'classnames.txt')

cl_names = jio.get('classnames.txt')
print("Class names: "+str(cl_names))
'''
for i in cl_names:
    print(i)
'''
# pr = model.predict_classes(im.reshape((1, 1, 28, 28)))  # for predicting for a single image

# print(train_generator.filenames)
# print(validation_generator.filenames)
