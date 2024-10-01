import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import warnings
import os
from tqdm import tqdm
import pickle

warnings.filterwarnings("ignore")

with open("features_list.pkl", "rb") as f:
    feature_embedding = pickle.load(f)

# Load the filenames file
with open("features_name.pkl", "rb") as f:
    filename = pickle.load(f)

print(len(feature_embedding))
print(len(filename))

#load the model
# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model
Model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])


#now load the image
img = image.load_img("test/1538.jpg", target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
features = Model.predict(img)
features = features.flatten()
result = features / np.linalg.norm(features).reshape(-1,1) #it gave us 1d array and we convert it into two 2d array.


neighbour = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbour.fit(feature_embedding)

distance,indices = neighbour.kneighbors(result)


print(indices)
for file in indices[0]:
    print(file)