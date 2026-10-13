# -*- coding: utf-8 -*-
"""
# Colorectal histology dataset with LeNet 5
Preview: Neural Network examples with `tf.keras`

# Colorectal histology
Methods to diagnose colorectal using histology images
In this case, the purpose is to classify the type of histology in a given image in the following categories:

- 0: TUMOR
- 1: STROMA
- 2: COMPLEX
- 3: LYMPHO
- 4: DEBRIS
- 5: MUCOSA
- 6: ADIPOSE
- 7: EMPTY
"""

#!pip install scikit-learn -U

"""## Import packages"""

# Temporary workaround for package version compatibility issues
!pip install --upgrade "protobuf>=6.31.1"
!pip install importlib_resources

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import metrics

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

print(tf.__version__)

"""## Define global constants

Lets start with a few epochs to test learning network parameters
"""

batch_size = 32
nb_classes = 2  #or 8
epochs = 25

# Scaling input image to theses dimensions
img_rows, img_cols = 32, 32

"""## Download image database and load function"""

def format_example(image):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (img_rows, img_cols))
    return image


def load_data(name="colorectal_histology", img_size=(150, 150)):
    # batch_size=-1 carga todos los datos directamente en un diccionario de tensores
    ds = tfds.load(name, split=tfds.Split.TRAIN, batch_size=-1)

    # Preprocesamiento vectorizado sobre el tensor 4D completo
    images = tf.cast(ds['image'], tf.float32) / 255.0
    images = tf.image.resize(images, img_size)

    # Conversión a NumPy (as_numpy ya devuelve ndarrays)
    return tfds.as_numpy(images), tfds.as_numpy(ds['label'])

"""## Plot images"""

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(25, 2))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i],:,:,:])
        plt.gray()
        ax.set_title(f'{y[index[i]]}-{index[i]}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

"""## Build LeNet5 structure

<center><img src="https://www.dlsi.ua.es/~juanra/UA/curso_verano_DL/images/LeNet5.jpg"></center>
"""

#
# Build an ANN structure - LeNet5
#

def cnn_model():
    #
    # Neural Network Structure
    #

    model = Sequential()

    model.add(layers.Conv2D(6, (5, 5), activation='sigmoid'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(16, (5, 5), activation='sigmoid'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation='sigmoid'))

    model.add(layers.Dense(84, activation='sigmoid'))

    model.add(layers.Dense(nb_classes, activation = 'softmax'))

    return model

"""## Start to run the program

### Load data
"""

##################################################################################
# Main program

X, y = load_data()

print(X.shape, 'train samples')
print(img_rows,'x', img_cols, 'image size')
print(epochs,'epochs')

"""Only for binary classification. All number of classes greater than 0 will be set to 1."""

if nb_classes==2:
  y[y>0] = 1

"""### Let to see examples of the dataset"""

plot_symbols(X, y, 15)

"""## Number of examples per class"""

pd.DataFrame({'ncount':pd.Series(y).value_counts(), '% ncount':pd.Series(y).value_counts(normalize=True)*100})

"""## Split examples in training/test sets

This section is introductory to serve as a simple example. To test the model created in different situations, a 10 cross validation (10-CV) strategy should be used.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f'X_train {X_train.shape} X_test {X_test.shape}')
print(f'y_train {y_train.shape} y_test {y_test.shape}')

"""### Model and optimizers

Test [optimizer](https://keras.io/optimizers/) parameter with `sgd`, `adadelta` or `adam` values in order to check the final precision achieved.
"""

# Compute the inverse class frequency
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print(class_weights)

model = cnn_model()

model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

# Entrenar pasando class_weight
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=2
)

print(model.summary())

loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f'loss: {loss:.2f} acc: {acc:.2f}')

# Output prediction for the first five X_test images
model.predict(X_test[:5], batch_size=batch_size)

"""## Prediction

### Testing AUC result for two and multiple classes
"""

y_scores = model.predict(X_test) # Confidence prediction per class
y_pred = y_scores.argmax(axis=1) # Select classes with most confidence prediction

if nb_classes ==  2:
  print(f'AUC {metrics.roc_auc_score(y_test, np.round(y_scores[:,1],2)):.4f} ')
else:
  print(f'AUC {metrics.roc_auc_score(y_test, y_scores, multi_class = "ovr"):.4f} ')

"""## More metrics about results

We can find more information about `precision`, `recall` and `f1` metrics in <https://en.wikipedia.org/wiki/Precision_and_recall>.
"""

print('Predictions', pd.Series(y_pred).value_counts(),'\n')

print('Confusion matrix')
print(metrics.confusion_matrix(y_test,y_pred),'\n')

target_names = ['TUMOR', 'HEALTHY'] if nb_classes ==  2 else ['TUMOR','STROMA','COMPLEX','LYMPHO','DEBRIS','MUCOSA','ADIPOSE','EMPTY']

print(metrics.classification_report(y_test, y_pred, target_names=target_names))
