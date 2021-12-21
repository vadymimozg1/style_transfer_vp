from flask import Flask, render_template, request, redirect, url_for, send_file, after_this_request, current_app
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
# import PIL.Image
import time
import functools

# import string
# import random
import tensorflow_hub as hub
import io
import base64



app = Flask(__name__)    

MODEL_FOLDER = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'


img_model =  hub.load(MODEL_FOLDER)
# img_model =  tf.saved_model.load(MODEL_FOLDER)

@app.route("/about")
def about_page():
	return "Developer: VM, Geneva, Switzerland, 12.21.2021. Deep Learning Web Application. Neural Style Transfer. Flask. HTML"

@app.route("/Sarah")
def message():
	return "Merry Christmas and Happy New Year 2022 Sarah!!!"



def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
    return Image.fromarray(tensor)





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

                imp_1 = Image.open(img1)
                obj_1 = io.BytesIO()             # file in memory to save image without using disk  #
                imp_1.save(obj_1, format='JPEG')  # save in file (BytesIO)                           # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
                encoded_img_data_1 = base64.b64encode(obj_1.getvalue())
                img_data_1=encoded_img_data_1.decode('utf-8') 


                imp_2 = Image.open(img2)
                obj_2 = io.BytesIO()             # file in memory to save image without using disk  #
                imp_2.save(obj_2, format='JPEG')  # save in file (BytesIO)                           # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
                encoded_img_data_2 = base64.b64encode(obj_2.getvalue())
                img_data_2=encoded_img_data_2.decode('utf-8') 

 
                
                gen_image = predict_label(img_path_content, img_path_style)

                # subname=id_generator(10, "+!#&%=_[]ABCDEfGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                
                # file_name_gen = 'my_image_' + subname+'.jpg'
                # tensor_to_image(gen_image).save("static/" + file_name_gen)
                # img_path_generated = "static/" + file_name_gen
 
                os.remove(img_path_content)
                os.remove(img_path_style)
                
                imp_3 = tensor_to_image(gen_image)
                obj_3 = io.BytesIO()             # file in memory to save image without using disk  #
                imp_3.save(obj_3, format='JPEG')  # save in file (BytesIO)                           # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
                encoded_img_data_3 = base64.b64encode(obj_3.getvalue())
                img_data_3=encoded_img_data_3.decode('utf-8')                 
                
 
            else: return redirect(url_for('main'))
        else: return redirect(url_for('main'))    

    return render_template("predict.html", img_data_1 = img_data_1, img_data_2 = img_data_2, img_data_3 = img_data_3)


#if __name__ =='__main__':
#	app.run()
