#!/usr/bin/env python3

import os
from datetime import datetime
from pymongo import MongoClient

from model import Model
from mtg_df_prep import load_color_data
from helpers import generate_flow_df, plot_conf_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg19 import VGG19  # 224x224 input
from tensorflow.keras.applications.inception_v3 import InceptionV3  # 299x299 input
from tensorflow.keras.applications.resnet50 import ResNet50  # 224x224 input
from tensorflow.keras.applications.xception import Xception  # 224x224 input

# Connect database
client = MongoClient('192.168.0.209', 27017)
db = client['capstone_3']
model_db = db['models']

# Load & filter data
cards = load_color_data(mono=True)
df = cards[cards.set_type.isin(['starter', 'core', 'expansion'])]
# df = cards[cards.set.isin(['7ed', '8ed', '9ed', 'm10', 'm12'])]

# Model params
model_dir = 'data/models'
img_dir = '/home/jovyan/data/art/'
target_size = (224, 224)
batch_size = 16
test_size = 0.2
epochs = 50

# Architectures

inception_base = InceptionV3(input_shape=target_size + (3,), include_top=False, weights='imagenet')
xception_base = Xception(input_shape=target_size + (3,), include_top=False, weights='imagenet')
resnet_base = ResNet50(input_shape=target_size + (3,), include_top=False, weights='imagenet')

for base_model in [inception_base, xception_base]:
    for layer in base_model.layers:
        layer.trainable = False


arch = [
    xception_base,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
]

arch1 = [ # best scratch model
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=target_size + (3,)),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(6, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(0.01))
]

note = None

def fit_model(df, note):
    model = Model(
        arch,
        db=model_db,
#         grayscale=True,
        data_dir=img_dir,
        target_size=target_size,
        batch_size=batch_size,
        test_size=test_size
    )

    model.fit(
        generate_flow_df(df),
        epochs=epochs
    )

    model.save(model_dir, note=note)

def main():
#     cards = load_color_data(mono=True, land=False)
#     df = cards[cards.set_type.isin(['starter', 'core', 'expansion'])]
#     note = 'Xception no lands, sets trained: {}'.format(df.set.unique())
#     fit_model(df, note)
    
    cards = load_color_data(mono=True, colorless=False)
    df = cards[cards.set_type.isin(['starter', 'core', 'expansion'])]
    note = 'Xception no colorless w/ dropout 50 epochs, sets trained: {}'.format(df.set.unique())
    fit_model(df, note)

if __name__ == '__main__':
    main()