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


def convert_grid_to_image(_cells):
    processed_img = np.zeros((28,28))
    for r in range(len(processed_img)):
        for c in range(len(processed_img[r])):
            processed_img[r][c] = _cells[r][c].luma
    reshaped_img = np.array(processed_img).reshape(-1,28,28)
    #print(reshaped_img)
    return reshaped_img


#image = cv2.imread("test_image_01.png", cv2.IMREAD_GRAYSCALE)  # uint8 image
#image = np.array(image).reshape(-1,28,28)
#image = image/255.0#normalize and convert to float32
#print(image)

#reds = get_prediction(image)
#print(preds[0])
#best_guess = np.argmax(preds)
#print('prediction: ', best_guess)
#plt.imshow(image[-1])#,cmap=plt.cm.binary)
#plt.xlabel('prediction: {}'.format(best_guess))
#
