from flask import Flask, render_template, request, redirect, url_for
import numpy as np

import matplotlib.pyplot as plt 

import keras 

import cv2 

import tensorflow

# example of loading an image with the Keras API

from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from tensorflow.keras.utils import load_img

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.applications.vgg16 import VGG16

import keras
import pickle
import tensorflow as tf
from tensorflow import keras
import werkzeug
import tensorflow as tf
import tensorflow_hub as hub
from werkzeug.wrappers import Response
import os
import time


# def call_predict():
    
   
    # image = load_img(IMAGE_PATH, target_size = (224,224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]
        
    # classification = '%s (2.2f%%)' % (label[1], label[2] *100)
    
    # return render_template("index.html")

    



app = Flask(__name__)

model = tf.keras.models.load_model('C://Users//Windows//Desktop//MEGATREND//PESEKI//model//12-16-2022 17:28PM-full-image-set-mobilenetv2-Adam3.h5', 
                                     custom_objects={"KerasLayer":hub.KerasLayer})

@app.route('/')
def success():
   return render_template("index.html")



@app.route("/", methods=["POST"])
def hello_world():
    
    imagefile = request.files["imagefile"]
    image_path = "./images/"+ imagefile.filename
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size = (224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    classification = '%s (2.2f%%)' % (label[1], label[2] *100)
    
    
    return render_template("second.html",prediction = classification)
    
    
    
  
    
    
    
     

    

if __name__ == '__main__':
    app.run(port=3000, debug = True, use_reloader=False)
