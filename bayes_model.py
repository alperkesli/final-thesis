import os
import cv2
import glob
import numpy as np
import pandas as pd

import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg

from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize


INPUT_PATH = "../Veriset"

print(os.listdir(INPUT_PATH))


train_normal = Path(INPUT_PATH + '/train/Normal').glob('*.jpeg')
train_pneumonia = Path(INPUT_PATH + '/train/Pneumonia').glob('*.jpeg')

normal_data = [(image, 0) for image in train_normal]
pneumonia_data = [(image, 1) for image in train_pneumonia]

train_data = normal_data + pneumonia_data

train_data = pd.DataFrame(train_data, columns=['image', 'label'])


train_data = train_data.sample(frac=1., random_state=100).reset_index(drop=True)


count_result = train_data['label'].value_counts()
print('Toplam : ', len(train_data))
print(count_result)

plt.figure(figsize=(8,5))
sns.countplot(x = 'label', data =  train_data)
plt.title('Sınıf Sayıları', fontsize=16)
plt.xlabel('Sınıf çeşidi', fontsize=14)
plt.ylabel('Miktar', fontsize=14)
plt.xticks(range(len(count_result.index)), 
           ['Normal : 0', 'Pnömoni : 1'], 
           fontsize=14)
plt.show()
   
    
im = Image.open(INPUT_PATH + '/train/Normal/IM-0115-0001.jpeg')
nor = np.array(im)
nor.resize(224,224)
print(nor.shape)



def load_data(files_dir='/train'):

    normal = Path(INPUT_PATH + files_dir + '/Normal').glob('*.jpeg')
    pneumonia = Path(INPUT_PATH + files_dir + '/Pneumonia').glob('*.jpeg')

    normal_data = [(image, 0) for image in normal]
    pneumonia_data = [(image, 1) for image in pneumonia]
    img_data = normal_data + pneumonia_data
    image_data = pd.DataFrame(img_data, columns=['image', 'label'])

    image_data = image_data.sample(frac=1., random_state=100).reset_index(drop=True)

    x_images, y_labels = ([data_input(image_data.iloc[i][:]) for i in range(len(image_data))], 
                             [image_data.iloc[i][1] for i in range(len(image_data))])

    x_images = np.array(x_images)
    x_images = x_images.reshape(x_images.shape[0],x_images.shape[1]*x_images.shape[2]*x_images.shape[3])
    
    y_labels = np.array(y_labels)
    
    return x_images,y_labels


def data_input(dataset):
    for image_file in dataset:
        image = cv2.imread(str(image_file))
        image = cv2.resize(image, (224,224))
        return image
    
    
x_train, y_train = load_data(files_dir='/train')

print(x_train.shape)
print(y_train.shape)


x_test, y_test = load_data(files_dir='/test')
print(x_test.shape)
print(y_test.shape)


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)


ypred = model.predict(x_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred) 

trainscore = model.score(x_train,y_train)
testscore = model.score(x_test,y_test)
print('Training score: {:.2f}\nTest score: {:.2f}'.format(trainscore*100,testscore*100))