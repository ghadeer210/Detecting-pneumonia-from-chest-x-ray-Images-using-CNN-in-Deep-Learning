# General libraries
import os
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
# Deep learning libraries
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications import VGG16

# Setting seeds for reproducibility
seed = 232
tf.random.set_seed(seed)
# --------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1843)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# ---------------------------

# input_path = '../input/chest_xray/chest_xray/'
input_path = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/'

plt.figure(1)
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()
for i, _set in enumerate(['train', 'val', 'test']):
    set_path = input_path + _set
    ax[i].imshow(plt.imread(set_path + '/NORMAL/' + os.listdir(set_path + '/NORMAL')[0]), cmap='gray')
    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
    ax[i + 3].imshow(plt.imread(set_path + '/PNEUMONIA/' + os.listdir(set_path + '/PNEUMONIA')[0]), cmap='gray')
    ax[i + 3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
plt.show()

# Distribution of our datasets
for _set in ['train', 'val', 'test']:
    n_normal = len(os.listdir(input_path + _set + '/NORMAL'))
    n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))
    print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))

# input_path = '../input/chest_xray/chest_xray/'
input_path = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/'


def process_data(img_dims, batch_size):
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # This is fed to the network in the specified batch sizes and image dimensions
    train_generator = train_datagen.flow_from_directory(
        input_path + 'train',
        target_size=img_dims,
        batch_size=batch_size,
        class_mode='categorical'
        # class_mode='binary'
    )
    test_generator = test_datagen.flow_from_directory(
        input_path + 'test',
        target_size=img_dims,
        batch_size=batch_size,
        class_mode='categorical'
        # class_mode='binary'
    )
    validation_generator = val_datagen.flow_from_directory(
        input_path + 'val',
        target_size=img_dims,
        batch_size=batch_size,
        class_mode='categorical'
        # class_mode='binary'
    )

    return train_generator, test_generator, validation_generator


# Hyperparameters
img_dims = 224
image_size = (img_dims, img_dims)
epochs = 20
batch_size = 32
num_classes = 2
# Getting the data
train_generator, test_generator, validation_generator = process_data(image_size, batch_size)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new fully connected layers on top of the pre-trained model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# Callbacks
checkpoint = ModelCheckpoint(
    filepath='C:/Users/Ghadeer/Desktop/projects/final-final/VGG16/tow neurial/best_weights.hdf5', save_best_only=True,
    save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min', verbose=0)
call = CSVLogger('C:/Users/Ghadeer/Desktop/projects/final-final/VGG16/tow neurial/mytrain.csv', separator=',',
                 append=False)
tensorboard = TensorBoard(log_dir='/CallBacks-Final-Final/VGG16', histogram_freq=1, write_graph=True, write_images=True)

# Fitting the model
history = model.fit(
    train_generator, steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs, validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[checkpoint, lr_reduce  # , early_stop
        , call, tensorboard])
# save model
model.save('C:/Users/Ghadeer/Desktop/projects/final-final/VGG16/tow neurial/arch.hdf5')
# Set the path to the saved model
model_path = 'C:/Users/Ghadeer/Desktop/projects/final-final/VGG16/tow neurial/arch.hdf5'
# Load the saved model
model = load_model(model_path)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
plt.show()
# --------------------------------------------------------------
# I will be making predictions off of the test set in one batch size
# This is useful to be able to get the confusion matrix

from sklearn.metrics import accuracy_score, confusion_matrix

preds = model.predict(test_generator)
preds = np.argmax(preds, axis=1)

acc = accuracy_score(test_generator.classes, preds) * 100
cm = confusion_matrix(test_generator.classes, preds)
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(10, 8), hide_ticks=True, show_absolute=True, show_normed=True,
                                cmap=plt.cm.Blues, colorbar=True)

print('\nTEST METRICS ----------------------')
precision = tp / (tp + fp) * 100
recall = tp / (tp + fn) * 100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2 * precision * recall / (precision + recall)))

print('\nTRAIN METRIC ----------------------')
print('Train accuracy: {}'.format(np.round((history.history['accuracy'][-1]) * 100, 2)))

# -----------------------------------------
# Evaluate the model on the train set
train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.samples // batch_size)
print('Train Loss:', train_loss)
print('Train Accuracy:', train_accuracy)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
# ----------------------------------------------------
# predict:
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

test_image_path1 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/NORMAL2-IM-1427-0001.jpeg'  # NORMAL
test_image_path2 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/person1954_bacteria_4886.jpeg'  # PNEUMONIA
# Load the saved model
# model = load_model('C:/Users/Ghadeer/Desktop/projects/final_ghadeer/Zero/arch.hdf5')

# Load and preprocess the test image
img = image.load_img(test_image_path2, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Perform inference on the test image
pred_probs = model.predict(img_array)
pred_label = np.argmax(pred_probs)
class_labels = list(test_generator.class_indices.keys())

# Print the predicted label
print('Predicted Class:', class_labels[pred_label])

# -----------------------------------------------------

from sklearn.metrics import accuracy_score, confusion_matrix

# Generate predictions for the test set
test_generator.reset()
pred_probs = model.predict(test_generator)

# Convert prediction probabilities to class labels
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Create the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(10, 8), hide_ticks=True, show_absolute=True, show_normed=True,
                                cmap=plt.cm.Blues, colorbar=True)

print('\nTEST METRICS ----------------------')
precision = tp / (tp + fp) * 100
recall = tp / (tp + fn) * 100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2 * precision * recall / (precision + recall)))

print('\nTRAIN METRIC ----------------------')
print('Train accuracy: {}'.format(np.round((history.history['accuracy'][-1]) * 100, 2)))

# ------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2292)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
