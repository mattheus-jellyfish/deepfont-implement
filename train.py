# from matplotlib.pyplot import imshow
# import matplotlib.cm as cm
# import matplotlib.pylab as plt
import os
import random
import PIL
from PIL import Image, ImageFilter
import cv2
import argparse
import itertools
import numpy as np
from imutils import paths
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks, optimizers
from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K
import glob

# Create a global font mapping
def create_font_mapping():
    font_files = sorted(glob.glob("fonts/*"))
    font_names = [os.path.splitext(os.path.basename(f))[0] for f in font_files]
    return {font: idx for idx, font in enumerate(font_names)}

# Global font mapping
FONT_MAPPING = create_font_mapping()

def pil_image(img_path):
    pil_im =PIL.Image.open(img_path).convert('L')
    pil_im=pil_im.resize((105,105))
    #imshow(np.asarray(pil_im))
    return pil_im

def noise_image(pil_im):
    img_array = np.asarray(pil_im)
    mean = 0.0   # some constant
    std = 5   # some constant (standard deviation)
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    noise_img = PIL.Image.fromarray(np.uint8(noisy_img_clipped)) # output
    #imshow((noisy_img_clipped ).astype(np.uint8))
    noise_img=noise_img.resize((105,105))
    return noise_img

def blur_image(pil_im):
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=3)) # ouput
    #imshow(blur_img)
    blur_img=blur_img.resize((105,105))
    return blur_img

def affine_rotation(img):
    #img=cv2.imread(img_path,0)
    rows, columns = img.shape

    point1 = np.float32([[10, 10], [30, 10], [10, 30]])
    point2 = np.float32([[20, 15], [40, 10], [20, 40]])

    A = cv2.getAffineTransform(point1, point2)

    output = cv2.warpAffine(img, A, (columns, rows))
    affine_img = PIL.Image.fromarray(np.uint8(output)) # affine rotated output
    #imshow(output)
    affine_img=affine_img.resize((105,105))
    return affine_img

def gradient_fill(image):
    #image=cv2.imread(img_path,0)
    laplacian = cv2.Laplacian(image,cv2.CV_64F)
    laplacian = cv2.resize(laplacian, (105, 105))
    return laplacian

def conv_label(label):
    # Return the index for the given label using the global mapping
    return FONT_MAPPING.get(label)

def create_model():
    model=Sequential()

    model.add(Conv2D(64, kernel_size=(48, 48), activation='relu', input_shape=(105,105,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(24, 24), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2DTranspose(128, (24,24), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(64, (12,12), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2383,activation='relu'))
    model.add(Dense(len(FONT_MAPPING), activation='softmax'))
  
    return model


def main(batch_size=128,epochs=25,data_path="train_data/"):
    data=[]
    labels=[]
    imagePaths = sorted(list(paths.list_images(data_path)))
    random.seed(42)
    random.shuffle(imagePaths)

    augument=["blur","noise","affine","gradient"]
    a=itertools.combinations(augument, 4)

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        label = conv_label(label)
        pil_img = pil_image(imagePath)
        #imshow(pil_img)
        
        org_img = img_to_array(pil_img)
        #print(org_img.shape)
        data.append(org_img)
        labels.append(label)
        
        augument=["noise","blur","affine","gradient"]
        for l in range(0,len(augument)):
        
            a=itertools.combinations(augument, l+1)

            for i in list(a): 
                combinations=list(i)
                temp_img = pil_img
                for j in combinations:
                
                    if j == 'noise':
                        temp_img = noise_image(temp_img)
                        
                    elif j == 'blur':
                        temp_img = blur_image(temp_img)
                        
                    elif j == 'affine':
                        open_cv_affine = np.array(pil_img)
                        temp_img = affine_rotation(open_cv_affine)

                    elif j == 'gradient':
                        open_cv_gradient = np.array(pil_img)
                        temp_img = gradient_fill(open_cv_gradient)
    
                temp_img = img_to_array(temp_img)
                data.append(temp_img)
                labels.append(label)


    data = np.asarray(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    # convert the labels from integers to vectors
    num_classes = len(FONT_MAPPING)
    trainY = to_categorical(trainY, num_classes=num_classes)
    testY = to_categorical(testY, num_classes=num_classes)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
    K.set_image_data_format('channels_last')

    model= create_model()
    sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    filepath="top_model.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [early_stopping,checkpoint]

    model.fit(trainX, 
        trainY,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(testX, testY),
        callbacks=callbacks_list)
    score = model.evaluate(testX, testY, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put training parameters')
    parser.add_argument('--epochs','-e',required=True)

    args = parser.parse_args()
    main(epochs=int(args.epochs))
