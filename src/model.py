#!/usr/bin/env python3


"""
    Description

     Parameters
    -------------------
        

     Returns
    -------------------
        
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime

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


def to_grayscale_then_rgb(image):
    """
        Creates grayscale image then resizes it to RGB shape
        for use with transfer learning models

         Parameters
        -------------------
            image (np.array): An array of image data

         Returns
        -------------------
            A grayscale image with RGB shape
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


class Model:
    def __init__(self, arch, db=None, grayscale=False, test_size=0.2, data_dir='./', target_size=(224,224), batch_size=8):
        self.arch = arch
        self.db = db
        self.grayscale = grayscale
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

        model = Sequential()

        for layer in self.arch:
            model.add(layer)

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
            preprocessing_function=to_grayscale_then_rgb if self.grayscale else None,
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
            y_col='target',
            target_size=self.target_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        
    def fit(self, data, epochs=5):
        """
            Split and fit data to the sequential model
            
             Parameters
            -------------------
                data : Pandas DataFrame to be fit
                epochs : Number of iteratioins of the learning process

             Returns
            -------------------
                Keras fit model history object
        """
        
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=19)
        self.flow_to_generators(train_data, test_data)
        
        self.history = self.model.fit(
            self.train_gen,
            validation_data=self.train_gen,
            steps_per_epoch=self.train_gen.n//self.train_gen.batch_size,
            validation_steps=self.valid_gen.n//self.valid_gen.batch_size,
            epochs=epochs
        )
        
        return self.history
    
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
        
        score = self.model.evaluate(self.valid_gen)
        return {
            'test_loss': score[0],
            'test_accuracy': score[1]
        }
    
    def predict(self, X, steps=None):
        """
            Calls the Keras model's predict method using the
            validation generator created in self.flow_to_generators()
            
             Parameters
            -------------------
                X : Data to be predicted
                steps : Number of steps to take if X is a generator object

             Returns
            -------------------
                A Prediction
        """

        return self.model.predict(X, steps=steps)
    
    def predict_proba(self, X, steps=None):
        """
            A Wrapper for the Keras model's predict_proba method
            
             Parameters
            -------------------
                X : Data to be predicted
                steps : Number of steps to take if X is a generator object

             Returns
            -------------------
                A set of probabilities for each class
        """

        return self.model.predict_proba(X, steps=steps)
    
    def predict_holdout(self):
        """
            Predicts on the holdout set held in the test generator
            created in self.flow_to_generators()
            
             Parameters
            -------------------
                None

             Returns
            -------------------
                A Pandas DataFrame of predictions on the holdout set
        """
        
        filenames = self.test_gen.filenames
        predict = self.predict(self.test_gen, len(filenames))
        ids = [f.split('/')[1].replace('.jpg','') for f in filenames]
        df = pd.DataFrame({
            'prediction': predict.argmax(axis=1),
            'true': self.test_gen.labels
        }, index=ids)

        return df
    
    def summary(self):
        """
            A wrapper for the built in model.summary() method

             Parameters
            -------------------
                None

             Returns
            -------------------
                A formatted model summary string
        """

        return self.model.summary()
    
    def save(self, filepath='./', dirname=None, note=None):
        """
            Saves model data to MongoDB and fit model to disk.

             Parameters
            -------------------
                filepath : the parent filepath to save the model directory 
                dirname : the directory that will hold the model - if None, will be a timestamp

             Returns
            -------------------
                None
        """

        created = datetime.now()
        dirname = dirname if dirname else str(round(created.timestamp()))
        fp =  os.path.join(filepath, dirname)
        hist = self.history.history

        self.db.insert_one({
            'eval_score': self.evaluate(),
            'epochs': len(hist) + 1,
            'history': hist,
            'holdout_prediction': self.predict_holdout().to_dict(),
            'created': datetime.now(),
            'model': fp,
            'note': note, 
            'class_indices': self.test_gen.class_indices
        })

        self.model.save(fp)


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
        Dense(6, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(0.01))
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
    print('Test loss:', score['test_loss'])
    print('Test accuracy:', score['test_accuracy'])
    

if __name__ == '__main__':
    main()