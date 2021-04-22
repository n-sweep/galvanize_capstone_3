#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

try:
    from mtg_df_prep import *
    from helpers import *
except:
    from src.mtg_df_prep import *
    from src.helpers import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
keras = tf.keras

color_key = {'CWUBRG'[i]: (i, c) for i, c in enumerate(['colorless','white','blue','black','red','green'])}


class Model:
    def __init__(self, arch, test_size=0.2, data_dir='./', target_size=(224,224), batch_size=8):
        self.arch = arch
        self.test_size = test_size
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size

        self.model = self.prepare_model()
        self.create_generators()

    def prepare_model(self):
        """
            Sets up model, adding layers and compiling
            
             Parameters
            -------------------
                arch (list): Architecture - list of Keras layers used to build the sequential model

             Returns
            -------------------
                A compiled Keras Sequential model
        """
        # TODO: add a default architecture?

        model = Sequential(self.arch)

        model.compile(  # TODO: feed in these parameters from outside the object?
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_generators(self):
        # Instantiate Image Data Generators
        self.train_idg = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.20
        )
        self.test_idg = ImageDataGenerator(rescale=1/255.0)
    
    def flow_to_generators(self, train_df, test_df):
        """
            Description
            
             Parameters
            -------------------
                train_df : 
                test_df : 

             Returns
            -------------------
                None
        """
    
        # Flow in filepaths from prepared DataFrame
        self.train_gen = self.train_idg.flow_from_dataframe(
            dataframe=train_df,
            directory = self.data_dir,
            x_col='path',
            y_col='target',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=19
        )

        self.valid_gen = self.train_idg.flow_from_dataframe(
            dataframe=train_df,
            directory=self.data_dir,
            x_col='path',
            y_col='target',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=19
        )

        self.test_gen = self.test_idg.flow_from_dataframe(
            dataframe=test_df,
            directory=self.data_dir,
            x_col='path',
            target_size=self.target_size,
            batch_size=1,
            class_mode=None,
            shuffle=False
        )
        
    def fit(self, data, epochs=5):
        """
            Description
            
             Parameters
            -------------------
                train_df : 
                test_df : 

             Returns
            -------------------
                None
        """
        
        train_data, test_data = train_test_split(data, test_size=self.test_size)
        self.flow_to_generators(train_data, test_data)
        
        history = self.model.fit(
            self.train_gen,
            validation_data=self.train_gen,
            steps_per_epoch=self.train_gen.n//self.train_gen.batch_size,
            validation_steps=self.valid_gen.n//self.valid_gen.batch_size,
            epochs=epochs
        )
        
        return history
    
    def evaluate(self):
        """
            Calls the fit Keras model's evaluate method using the
            validation generator created in self.flow_to_generators()
            
             Parameters
            -------------------
                None

             Returns
            -------------------
                None
        """
        return self.model.evaluate(self.valid_gen)


def main():
    target_size = (224, 224)
    batch_size = 8

    arch = [
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=target_size + (3,)),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128, (3,3), activation='relu'),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2,2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(l=0.01))
    ]

    model = Model(
        arch,
        test_size=0.2,
        data_dir='/home/jovyan/data/art/',
        target_size=target_size,
        batch_size=batch_size
    )

    cards = load_color_data(mono=True)
    cards = cards[cards.set_type.isin(['core'])]

    model.fit(generate_flow_df(cards), epochs=20)

    score = model.evaluate()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    

if __name__ == '__main__':
    main()