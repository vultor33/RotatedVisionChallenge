from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras.layers import GlobalAveragePooling2D
from keras.models import model_from_json
import collections
import os
import cv2
import pandas as pd
import numpy as np
import random


NOT_ROTATE = -1
NUMBER_CLASSES = 4
DATA_FOLDER = 'test'




def main():
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'rotvision_trained_model.h5'

    # load json and create model
    model_json_path = os.path.join(save_dir, model_name.split('.')[0] + '.json')
    model_path = os.path.join(save_dir, model_name)

    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)

    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)  # 2.2.0 to 2.3.1  --> lr changed to learning_rate
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])    

    print("Loaded model from disk")


    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'rotvision_trained_model.h5'

    # load json and create model
    model_json_path = os.path.join(save_dir, model_name.split('.')[0] + '.json')
    model_path = os.path.join(save_dir, model_name)

    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)

    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)  # 2.2.0 to 2.3.1  --> lr changed to learning_rate
    loaded_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])    

    print("Loaded model from disk")

    test_files = []
    for root, dirs, files in os.walk('test'):
        test_files += files
    all_files = test_files
    batch_size=32
    number_batches = int(len(all_files) / batch_size) + 1
    pred_labels = []
    corrected_images = []
    for i in range(number_batches):
        start = i* batch_size
        end = (i + 1) * batch_size

        file_list = all_files[start:end]
        if len(file_list) == 0:
            break
        results = write_orientated_images(file_list, loaded_model, 'ziptest')
        pred_labels += results[0]
        corrected_images.append(results[1])

    corrected_images = np.array(corrected_images)

    test_pred = {}
    test_pred['fn'] = all_files
    test_pred['label'] = pred_labels
    pd.DataFrame(test_pred).to_csv('test.preds.csv', index=False)
    np.save('test_array.npy', corrected_images)



def rot_corrections(label_code):
    if label_code == 0:
        return cv2.ROTATE_90_CLOCKWISE
    elif label_code == 1:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    elif label_code == 2:
        return NOT_ROTATE
    elif label_code == 3:
        return cv2.ROTATE_180

def correct_image(image, rot_correction):
    if rot_correction == NOT_ROTATE:
        return image
    else:
        return cv2.rotate(image, rot_correction)    

def code_tolabels(label_code):    
    if label_code == 0:
        return 'rotated_left'
    elif label_code == 1:
        return 'rotated_right'
    elif label_code == 2:
        return 'upright'
    elif label_code == 3:
        return 'upside_down'

def labels_toonehot(label):
    if label == 'rotated_left':
        return 0
    elif label == 'rotated_right':
        return 1
    elif label == 'upright':
        return 2
    elif label == 'upside_down':
        return 3
        
        
def load_data(file_list, to_preprocess = True):
    data = []
    for file in file_list:
        file_path = os.path.join(DATA_FOLDER, file)
        image = cv2.imread(file_path)
        
        if to_preprocess:
            image = preprocess(image)
        data.append(image)

    data = np.array(data)
    return data

def preprocess(image):
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    image = image.astype('float32')
    image /= 255
    return image
    


def prediction(file_list, model):
    data_xi = load_data(file_list)
    ypred_i = model.predict(data_xi)
    ypred_i = np.argmax(ypred_i, axis=1)
    return ypred_i

def write_orientated_images(file_list, model, folder = ''):
    file_dir = os.path.join(os.getcwd(), folder)    
    
    original_images = load_data(file_list, False)
    ypred_i = prediction(file_list, model)
    corrections = [rot_corrections(x) for x in ypred_i]
    file_names = [x.split('.')[0] for x in file_list]
    corrected_images = []
    
    for file_name, image, correction in zip(file_list, original_images, corrections):
        image_corr = correct_image(image, correction)
        corrected_images.append(image_corr)
        file_name = os.path.join(file_dir, file_name)    
        cv2.imwrite(file_name + '.png', image_corr)
    
    results = [
        [code_tolabels(x) for x in ypred_i],
        np.array(corrected_images)]
    return results        
    
    
    
if __name__ == "__main__":
    main()
    