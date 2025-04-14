import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD


# Define the autoencoder for unsupervised pretraining
def create_autoencoder():
    autoencoder = Sequential()
    
    # Encoder
    autoencoder.add(Conv2D(64, kernel_size=(11,11), strides=2, activation='relu', 
                          padding='valid', input_shape=(105,105,1)))
    autoencoder.add(MaxPooling2D(pool_size=(2,2)))
    autoencoder.add(Conv2D(128, kernel_size=(1,1), strides=1, activation='relu', padding='same'))
    
    # Decoder
    autoencoder.add(Conv2DTranspose(64, kernel_size=(1,1), strides=1, activation='relu', padding='same'))
    autoencoder.add(UpSampling2D(size=(2,2)))
    autoencoder.add(Conv2DTranspose(1, kernel_size=(11,11), strides=2, activation='relu', padding='valid'))

    # Compile autoencoder
    sgd = SGD(learning_rate=0.01)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
    
    return autoencoder


# Define CNN binary classifier (to be used after autoencoder pretraining)
def create_binary_classifier(pretrained_autoencoder=None):
    cnn = Sequential()
    
    # Cu layers - Feature extraction layers transferred from autoencoder
    cnn.add(Conv2D(64, kernel_size=(11,11), strides=2, activation='relu', 
                  padding='valid', input_shape=(105,105,1)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    cnn.add(Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding='valid'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    # Transfer weights from autoencoder if provided
    if pretrained_autoencoder is not None:
        cnn.layers[0].set_weights(pretrained_autoencoder.layers[0].get_weights())
        cnn.layers[3].set_weights(pretrained_autoencoder.layers[2].get_weights())
        
        # Freeze the transferred layers
        cnn.layers[0].trainable = False
        cnn.layers[3].trainable = False
    
    # Cs layers - Classification specific layers
    cnn.add(Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same'))
    cnn.add(Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same'))
    cnn.add(Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(4096, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(4096, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, activation='sigmoid'))  # Binary output - Denton Light or Not
    
    return cnn


# Custom metrics for binary classification
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_score = true_positives / (possible_positives + K.epsilon())
    return recall_score

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_score = true_positives / (predicted_positives + K.epsilon())
    return precision_score

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


# Function to compile the binary classifier
def compile_binary_model(model):
    sgd = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
    model.compile(
        optimizer=sgd,
        loss='binary_crossentropy',
        metrics=['accuracy', precision, recall, f1, tf.keras.metrics.AUC()]
    )
    return model 