import streamlit as st
import os
from test import neighbour
st.header("Fashion Recommendation System")
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
from PIL import Image

warnings.filterwarnings("ignore")

#file upload
#load_file / save file
#recommendation
#show the result

#load the embedding
with open("features_list.pkl", "rb") as f:
    feature_embedding = pickle.load(f)

# Load the filenames file
with open("features_name.pkl", "rb") as f:
    filename = pickle.load(f)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model
Model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])


def save_upload_file(file_uploaded):
    try:
        with open(os.path.join("uploads", file_uploaded.name), "wb") as code:
            code.write(file_uploaded.getbuffer())
        return 1
    except:
        return 0
def feature_extraction(uploaded_file, Model):   # For feature extraction from the uploaded image
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  #
    features = Model.predict(img_array)
    features = features.flatten()
    result = features / np.linalg.norm(features)
    return result

def recommendations(features,feature_embedding): #for recommendation
    neighbour = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbour.fit(feature_embedding)

    distances,indices = neighbour.kneighbors(features) #it will find the similar feature for image that was upload from the embedding that was fit into the model NearstNeighbour
    return indices




#Web______starting from here
file_uploader = st.file_uploader("Choose image from gallery")

if file_uploader is not None:
    if save_upload_file(file_uploader):
        image = Image.open(file_uploader)
        st.image(image)
        #call feature extraction for image
        features = feature_extraction(file_uploader, Model) #by this we will get the features of image
        #reshape the feature it should have the shape of 2D.
        features = features.reshape(1, -1)

        # Assuming 'filename' contains paths to the images that correspond to the features in feature_embedding
        indices = recommendations(features, feature_embedding)

        # Show the similar result
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        for i in range(6):
            with eval(f"col{i + 1}"):
                # Get the index for the similar images
                img_index = indices[0][i]  # Retrieve the index of the similar image
                img_path = filename[img_index]  # Get the actual path of the image
                img = Image.open(img_path)  # Open the image
                st.image(img, caption=f"Similar Image {i + 1}")  # Display the image with a caption

    else:
        print("Some error has been occcured.")