#this is to setup the environmental issue in laptop and may not occur in everyone's PC
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
#Project
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (224, 224))  
    normalized = resized / 255.0  
    return normalized

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def predict_floors(model, image_path):
    processed_image = preprocess_image(image_path)
    image_input = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(image_input)
    return int(prediction[0][0])

def batch_process(model, image_paths):
    results = {}
    for img_path in image_paths:
        num_floors = predict_floors(model, img_path)
        results[img_path] = num_floors
    return results


def display_result(image_path, num_floors):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(f'Number of Floors: {num_floors}')
    plt.axis('off')  
    plt.show()


if __name__ == '__main__':
    
    model = create_model()
    image_path = 'image7.jpg' #u need to enter the image of building 
    num_floors = predict_floors(model, image_path)
    print(f'Predicted number of floors: {num_floors}')
    display_result(image_path, num_floors)
    image_paths = ['image5.jpg','image6.jpg','image8.jpg','image9.png','photo.png','image1.png']
    batch_results = batch_process(model,image_paths)
    for img, floors in batch_results.items():
        print(f'{img}: {floors} floors')
        display_result(img, floors)
