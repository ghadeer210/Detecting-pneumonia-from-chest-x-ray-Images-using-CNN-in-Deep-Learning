import tensorflow as tf
from keras import Model
x = 5
import cv2
import os  
import matplotlib.pyplot as plt 
import numpy as np

os.chdir(os.path.join('c:/Users/Ghadeer/Desktop/Jobs/كسف تزوير الصور/DataSet/real_and_fake_face/', 'training_fake'))
child= cv2.imread('easy_1_1110.jpg',1)