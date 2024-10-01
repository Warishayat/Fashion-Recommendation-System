import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import warnings
import os
from tqdm import tqdm
import pickle

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model
Model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])

# Function to extract features from an image
def extract_features(image_path, Model):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = Model.predict(img)
    features = features.flatten()
    features = features / np.linalg.norm(features)  # Normalize features

    return features

filename = []  #to get the file names that we have
for file in os.listdir("images"):
    filename.append(os.path.join("images", file))

features_list = []  #feature list or vector

for file in tqdm(filename):
    features_list.append(extract_features(file, Model))

print(np.array(features_list).shape)

# Save features_list to a pickle file
pickle.dump(filename,open('features_name.pkl', 'wb'))
pickle.dump(features_list,open('features_list.pkl', 'wb'))
