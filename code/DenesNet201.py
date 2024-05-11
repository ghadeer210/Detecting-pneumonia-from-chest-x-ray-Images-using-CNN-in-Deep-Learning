# General libraries
import os
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.densenet import DenseNet201
# Setting seeds for reproducibility
seed = 232
tf.random.set_seed(seed)
#------------------------------------

# input_path = '../input/chest_xray/chest_xray/'
input_path = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/'

plt.figure(1)
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()
for i, _set in enumerate(['train', 'val', 'test']):
    set_path = input_path+_set
    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')
    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')
    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
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
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # This is fed to the network in the specified batch sizes and image dimensions
    train_generator = train_datagen.flow_from_directory(
        input_path + 'train', 
        target_size=img_dims, 
        batch_size=batch_size,
        #class_mode='categorical'
        class_mode='binary'
        )
    test_generator = test_datagen.flow_from_directory(
        input_path + 'test', 
        target_size=img_dims, 
        batch_size=batch_size, 
        #class_mode='categorical'
        class_mode='binary'
        )
    validation_generator = val_datagen.flow_from_directory(
        input_path +'val',
        target_size=img_dims,
        batch_size=batch_size,
        #class_mode='categorical'
        class_mode='binary'
        )


    return train_generator, test_generator, validation_generator

# Hyperparameters
img_dims = 224
image_size = (img_dims, img_dims)
epochs = 20
batch_size = 32
num_classes = 1
# Getting the data
train_generator, test_generator, validation_generator = process_data(image_size, batch_size)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new fully connected layers on top of the pre-trained model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(filepath='C:/Users/Ghadeer/Desktop/projects/final-final/DenesNet201/one neurial/best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min', verbose=0)
call = CSVLogger('C:/Users/Ghadeer/Desktop/projects/final-final/DenesNet201/one neurial/mytrain.csv', separator=',', append=False)
tensorboard = TensorBoard(log_dir='/CallBacks-Final-Final/DenesNet201', histogram_freq=1, write_graph=True, write_images=True)


#Fitting the model
history = model.fit(
                    train_generator, steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs, validation_data=test_generator,
                    validation_steps=test_generator.samples // batch_size, 
                    callbacks=[checkpoint, lr_reduce#, early_stop
                               , call, tensorboard])
#save model
model.save('C:/Users/Ghadeer/Desktop/projects/final-final/DenesNet201/one neurial/arch.hdf5')
# Set the path to the saved model
model_path = 'C:/Users/Ghadeer/Desktop/projects/final-final/DenesNet201/one neurial/arch.hdf5'
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
#--------------------------------------------------------------
# I will be making predictions off of the test set in one batch size
# This is useful to be able to get the confusion matrix
test_data = []
test_labels = []

for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
        
test_data = np.array(test_data)
test_labels = np.array(test_labels)



from sklearn.metrics import accuracy_score, confusion_matrix

preds = model.predict(test_data)

acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(10,8), hide_ticks=True,show_absolute=True,show_normed=True,cmap=plt.cm.Blues,colorbar=True)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train accuracy: {}'.format(np.round((history.history['accuracy'][-1])*100, 2)))

#-----------------------------------------
print(model.evaluate(train_generator))


#predict:
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

test_image_path1 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/NORMAL2-IM-1440-0001.jpeg' #NORMAL
test_image_path2 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/person1946_bacteria_4874.jpeg' #PNEUMONIA
# Load the saved model
#model = load_model('C:/Users/Ghadeer/Desktop/projects/final_ghadeer/Zero/arch.hdf5')

# Load and preprocess the test image
img = image.load_img(test_image_path1, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Perform inference on the test image
pred_probs = model.predict(img_array)
pred_label = np.argmax(pred_probs)
class_labels = list(test_generator.class_indices.keys())

# Print the predicted label
print('Predicted Class:', class_labels[pred_label])

#-----------------------------------------------------
print(model.evaluate(train_generator))


import numpy as np
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import load_img
test_image_path1 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/NORMAL2-IM-1440-0001.jpeg' #NORMAL
test_image_path2 = 'C:/Users/Ghadeer/Desktop/projects/final-final/chest_xray/person1946_bacteria_4874.jpeg' #PNEUMONIA
test_image = image.load_img(test_image_path1, target_size=(224,224,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image /255.0
result = model.predict(test_image)
result = ((result > 0.5)+0).ravel()
print(train_generator.class_indices)
if result==0:
    prediction='NORMAL'
else:
    prediction='PNEUMONIA'
print('The prediction result is :', prediction,  result)
