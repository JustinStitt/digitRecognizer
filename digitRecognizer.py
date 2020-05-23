import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import image
filename = 'digitReader_model.model'

#load our saved model
loaded_model = tf.keras.models.load_model(filename)

#make predictions
def get_prediction(data_point):
    #data_point.shape = (1,28,28)
    #print(data_point.shape)
    pred = loaded_model.predict(data_point)
    return pred

image = cv2.imread("test_image_01.png", cv2.IMREAD_GRAYSCALE)  # uint8 image
image = np.array(image).reshape(-1,28,28)
image = image/255.0#normalize and convert to float32
#print(image)

preds = get_prediction(image)
print(preds[0])
print(np.argmax(preds))
plt.imshow(image[-1],cmap=plt.cm.binary)
plt.show()
