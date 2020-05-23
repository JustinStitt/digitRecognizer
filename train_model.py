#using Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
#mnist dataset
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()


#print(y_train[0])
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#normalize our features (0-255) to (0-1)
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)
#build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())#serve as input layer (we take 28x28 and flatten to 1x784)

model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))#hidden layer with 256 nodes Densely connected
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))#hidden layer with 256 nodes Densely connected

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))#output layer. 10 possibilities (0,1,2,3,4,5,6,7,8,9)

#compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#adam is great default optimzier function to start with. same with relu for hidden layers

#Next, we have our loss metric. Loss is a calculation of error.
#A neural network doesn't actually attempt to maximize accuracy. It attempts to minimize loss.
#Again, there are many choices, but some form of categorical crossentropy
#is a good start for a classification task like this.

#now we fit (train)
model.fit(x_train,y_train,batch_size = 64, validation_split = 0.4, epochs = 3)#epochs = iterations

#test on data we didnt use in the model
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)
#save our model
#filename = 'digitReader_model.model'
#model.save(filename)


#to load a model
#loaded_model = tf.keras.models.load_model(filename)

#finally... make predictions!
#preds = model.predict(x_test)
#this will yield a distribution of probability for each ouput. lets take the max value (most likely as determined by our model)
#print(np.argmax(preds[0]))#for index 0, the highest prob from [0-9] is our prediction
#plt.imshow(x_test[0],cmap=plt.cm.binary)
#plt.show()


#added
image = cv2.imread("test_image_01.png", cv2.IMREAD_GRAYSCALE)  # uint8 image
image = np.array(image).reshape(-1,28,28)
image = image/255.0#normalize and convert to float32
#print(image)

preds = model.predict(image)
print(preds[0])
print(np.argmax(preds))
plt.imshow(image[-1],cmap=plt.cm.binary)
plt.show()
