

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import tensorflow as tf
import keras



VGG16_MODEL=tf.keras.applications.VGG16(include_top=True,
                                               weights='imagenet')


img = image.load_img("C:/Users/sruja/Desktop/Courses/Deep Learning/Ex_Files_Deep_Learning_Image_Recog_Upd/Exercise Files/Ch05/hamster.jpg", target_size=(224, 224,3))
plt.imshow(img)



# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a fourth dimension (since Keras expects a list of images)
x = np.expand_dims(x, axis=0)

# Normalize the input image's pixel values to the range used when training the neural network
x = tf.keras.applications.vgg16.preprocess_input(x)
# Run the image through the deep neural network to make a prediction
predictions =VGG16_MODEL.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = tf.keras.applications.vgg16.decode_predictions(predictions)

print("Below image is a:",predicted_classes[0][0][1])
