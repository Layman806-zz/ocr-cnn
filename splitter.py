import numpy as np
import cv2
from PIL import Image
import os

directory = 'TrainingData'
out_path_train = 'Train'
out_path_test = 'Test'

if not os.path.exists(out_path_train):
    os.makedirs(out_path_train)

if not os.path.exists(out_path_test):
    os.makedirs(out_path_test)

for i in range(62):
    c_path = directory + '/' + str(i+1)
    for j in range(1016):
        f_path = c_path + '/' + str(j) + '.png'

        im = Image.open(f_path)

        if j < 916:
            if not os.path.exists('Train/'+str(i)):
                os.makedirs('Train/'+str(i))
            im.save('Train/'+str(i)+'/'+str(j)+'.png')
        else:
            if not os.path.exists('Test/'+str(i)):
                os.makedirs('Test/'+str(i))
            im.save('Test/'+str(i)+'/'+str(j)+'.png')


