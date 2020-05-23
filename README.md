# digitRecognizer

Regarded as the "Hello, World!" of machine learning.

Uses Tensorflow and Keras with MNIST dataset to recognize handwritten digits with reasonable accuracy.

I've trained the model using a feed-forward sequential deep learning neural network.

The model trained on 40,000 feature sets with labels for the digits (0-9).
Then the model was tested for minimal loss and high accuracy against another 10,000 unseen feature sets. 
An accuracy of about 97% is what I achieved and is quite easy to achieve.

The next step would be to upgrade to a CNN which is notoriously better for computer vision AI tasks.

**Example w/ Interface for Drawing Digits**

![](digitsRecogGIF.gif)

(nerdy stuff)

dataset used (MNIST) = http://yann.lecun.com/exdb/mnist/

optimizer = "adam"
loss = "sparse_categorical_crossentropy"

model layout:

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())#serve as input layer (we take 28x28 and flatten to 1x784)

model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))#hidden layer with 256 nodes Densely connected
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))#hidden layer with 256 nodes Densely connected

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))#output layer. 10 possibilities (0,1,2,3,4,5,6,7,8,9)

**good resources**

https://en.wikipedia.org/wiki/Feedforward_neural_network
https://www.youtube.com/watch?v=IHZwWFHWa-w (3b1b)
