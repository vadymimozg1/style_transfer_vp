from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import tensorflow as tf 
import tarfile
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image
import time
import functools

import string
import random



app = Flask(__name__)    



#MODEL_FOLDER = 'C:/Users/mozgo/Documents/Studies/Spec_Cloud/course_2/my_application_v4'

MODEL_FOLDER = ''


img_model =  tf.saved_model.load(MODEL_FOLDER)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def predict_label(img_path_content, img_path_style):
    content_image = load_img(img_path_content)
    style_image = load_img(img_path_style)
    stylized_image = img_model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image
                
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "oups"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output1():
    if (request.method == 'POST'):    
        img1 = request.files['my_image1']
        img2 = request.files['my_image2']
        if (img1.filename != ''):
            if (img2.filename != ''):          
                img_path_content = "static/" + img1.filename
                img_path_style = "static/" + img2.filename
                
                img1.save(img_path_content)
                img2.save(img_path_style)
                
                gen_image = predict_label(img_path_content, img_path_style)
                subname=id_generator(10, "+!#&%=_[]ABCDEfGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                
                file_name_gen = 'my_image_' + subname+'.jpg'
                tensor_to_image(gen_image).save("static/" + file_name_gen)
                img_path_generated = "static/" + file_name_gen
                
            else: return redirect(url_for('main'))
        else: return redirect(url_for('main'))    

    return render_template("predict.html", img_path_content = img_path_content, img_path_style = img_path_style, img_path_generated = img_path_generated, file_name_gen=file_name_gen)


if __name__ =='__main__':
	#app.debug = True
	app.run()

# debug = True
