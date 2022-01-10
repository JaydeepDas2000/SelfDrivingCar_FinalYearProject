import pandas as pd # for Dataframe
import numpy as np # for conversion into array
import matplotlib.pyplot as plt # for graphical representation
import os # file path
from sklearn.utils import shuffle # data shuffling
import matplotlib.image as mpimg # import image
from imgaug import augmenters as iaa # image augmentation
import cv2 # image operation
import random # randomness

# for creating CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



def getName(filePath):      ## Remove image data path
    return filePath.split('\\')[-1]

def importDataInfo(path):   ## Import data from myData folder
    coloums =  ['Center','Left','Right','Steering','Throttle','Break','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'), names = coloums)
    data['Center'] = data['Center'].apply(getName)
    print('Total Image Imported : ',data.shape[0])
    return data

# Visualization and Distrubation of Data

def balanceData(data, display=True):
    nBins = 31
    samplePerBin = 3000 # for Keyboard user
    hist, bins = np.histogram(data['Steering'],nBins)
    print(bins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        print(center)
        plt.bar(center,hist,width=0.06)
        # plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show

    # remove the extra data
    # for Keyboard user
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range (len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList) ## it will help to shuffle the value
        binDataList = binDataList[samplePerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images : ', len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace=True) ## now we have to remove those indexes from orginal data
    print('Remaining Images : ', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'],nBins)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show
    
    return data

def loadData(path,data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexData = data.iloc[i]
        #print(indexData)
        imagesPath.append(os.path.join(path,'IMG',indexData[0]))
        #print(os.path.join(path,'IMG',indexData[0]))
        steering.append(float(indexData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    
    return imagesPath,steering

# Image Augmentation (Step 6)

def augmentImage(ImgPath, steering):
    img = mpimg.imread(ImgPath)

    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering

    return img, steering


# imgRe, st = augmentImage('test.jpg',0)
# plt.imshow(imgRe)
# plt.show()

# Pre-processing of Images (Step 7)
def preProcessing(img):
    img = img[60:135,:,:] # CROP
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # RGB -> YUV
    img = cv2.GaussianBlur(img,(3,3),0) # BLUR
    img = cv2.resize(img,(200,66)) # RESIZE
    img = img/255 # NORMALIZATION

    return img

# imgRe = preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgRe)
# plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img , steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))

# NVIDIA Model

def createModel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')

    return model
