import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfl

import matplotlib.pyplot as plt

import numpy as np

batch_size=32
img_height=256
img_width=256

train_dir='/Users/aditkadakia/Desktop/csci1470/fresh_tea_leaves/data'

train_ds=tfk.utils.image_dataset_from_directory(
    train_dir,
    seed=42,
    validation_split=0.1,
    subset='training',
    image_size=(img_height,img_width),
    batch_size=batch_size
)

val_ds=tfk.utils.image_dataset_from_directory(
    train_dir,
    seed=42,
    validation_split=0.1,
    subset='validation',
    image_size=(img_height,img_width),
    batch_size=batch_size
)

