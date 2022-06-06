# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Dataset
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

print(train_images.shape)  # So we've got 60,000 images that are made up of 28x28 pixels (784 in total).
print(train_images[0, 23, 23])  # let's have a look at one pixel
# we have a grayscale image as there are no color channels.
train_labels[:10]  # let's have a look at the first 10 training labels
# we have 10 different classes ( clothing brands )
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# let's see 1 of these images
plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()

# Data Preprocessing
# (scuishing or scaling) we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the Model ( architecture )
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2) # Activation Function
    keras.layers.Dense(10, activation='softmax')  # output layer (3)
])

# Compile the Model
model.compile(optimizer='adam',  # Gradient Ascend
              loss='sparse_categorical_crossentropy',  # Loss function
              metrics=['accuracy'])  # the output we want to see

# Training the Model
model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs

# Evaluating the Model ( findng the real accuracy )
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)  # Overfitting : accuracy is lower than the ones at training !!!

# Making Predictions
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])
print(test_labels[0])

# Verifying Predictions

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.label-color'] = COLOR


def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

