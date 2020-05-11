import tensorflow as tf
class callbackstop(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,check={}):
    if(check.get("accuracy")>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training= True
mystop=callbackstop()
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

x_train = x_train/255
x_test =x_test/255

model = tf.keras.models.Sequential([

tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5, callbacks=[mystop] )
model.evaluate(x_test,y_test)
predict = model.predict(x_test)
print(predict[1])
print(y_test[1])
plt.imshow(x_test[1])
!pip install --upgrade pip

from PIL import Image
import numpy as np
img = np.invert(Image.open("pic.png").convert("L")).ravel()

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))
