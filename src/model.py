import os, sys
import numpy as np
import pandas as pd

from src.util import Logger
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
keras = tf.keras

color_key = {'CWUBRG'[i]: (i, c) for i, c in enumerate(['colorless','white','blue','black','red','green'])}


def remove_lands(cards):
    """
        Remove lands from a DataFrame of Scryfall.com card data
        
         Parameters
        -------------------
            cards : A Pandas DataFrame of mtg card data

         Returns
        -------------------
            A Pandas DataFrame of mtg card data with the lands removed
    """
    mask = cards['type_line'].apply(lambda x: 'Land' in x)
    return cards[~mask]

def mono_color_cards(cards):
    """
        Take in a DataFrame of card info from ScryFall.com and returns
        only the mono-colored cards
        
         Parameters
        -------------------
            cards: A Pandas DataFrame of mtg card data
            
         Returns
        -------------------
            A Pandas DataFrame of mtg card data for only mono-color cards
    """
    mask = cards['colors'].apply(lambda x: len(x) < 2 if type(x) == list else False)
    return cards[mask]

def generate_flow_df(cards):
    """
        Creates a Pandas Dataframe of paths and targets to be
        read by tensorflow
        
         Parameters
        -------------------
            cards : A Pandas DataFrame of mtg card data

         Returns
        -------------------
            A Pandas Dataframe of paths and targets
    """
    paths = [f"{card[0]}/{card[1]}.jpg" for card in cards[['set', 'id']].values]
    targets = cards['colors'].apply(lambda x: x[0] if x else 'C')
    return pd.DataFrame({'path': paths, 'target': targets})

def prepare_data(df, test_size=0.25, mono=True, lands=False):
    """
        Remove multicolored and/or lands from the dataframe
        & do a train/test split

         Parameters
        -------------------
            df : A Pandas DataFrame of card data from Scryfall.com
            mono (bool) : Whether to remove multi-colored cards
            lands (bool) : Whether to remove lands
         Returns
        -------------------
            train & test Pandas DataFrames
    """
    
    df = mono_color_cards(df) if mono else df
    df = df if lands else remove_lands(df)
    df = generate_flow_df(df)
    train, test = train_test_split(df, test_size=test_size)
    
    return train, test

def prepare_model(input_shape):
    """
        Sets up model, adding layers and compiling
        
         Parameters
        -------------------
            target_size (tup): A tuple containing the size of the input images

         Returns
        -------------------
            A compiled Keras Sequential model
    """
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_regularizer =tf.keras.regularizers.l1( l=0.01)))
    model.add(Dense(6, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_generators():
    # Instantiate Image Data Generators
    train = ImageDataGenerator(
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
    test = ImageDataGenerator(rescale=1/255.0)
    
    return train, test

def flow_to_generators(train_df, test_df, parent_dir='./', target_size=(128, 128), batch_size=8):
    
    train_idg, test_idg = create_generators()
    
    # Flow in filepaths from prepared DataFrame
    train_gen = train_idg.flow_from_dataframe(
        dataframe=train_df,
        directory = parent_dir,
        x_col='path',
        y_col='target',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=19
    )

    valid_gen = train_idg.flow_from_dataframe(
        dataframe=train_df,
        directory=parent_dir,
        x_col='path',
        y_col='target',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=19
    )

    test_gen = test_idg.flow_from_dataframe(
        dataframe=test_df,
        directory=parent_dir,
        x_col='path',
        target_size=target_size,
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    
    return train_gen, valid_gen, test_gen

def main():
    parent_dir = '/home/jovyan/data/art/'
    target_size = (224, 224)
    batch_size = 8

    # Prep and split dataset
    cards = pd.read_json('data/cards.json')
    cards = cards[cards['set'].isin(['m10'])]
    
    train_df, test_df = prepare_data(cards, test_size=0.2)

    train_gen, valid_gen, test_gen = flow_to_generators(
        train_df, test_df,
        parent_dir=parent_dir,
        target_size=target_size,
        batch_size=batch_size
    )
    
    model = prepare_model(target_size + (3,))
    history = model.fit(
        train_gen,
        validation_data=train_gen,
        steps_per_epoch=train_gen.n//train_gen.batch_size,
        validation_steps=valid_gen.n//valid_gen.batch_size,
        epochs=10
    )
    
    score = model.evaluate(valid_gen)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    

if __name__ == '__main__':
    sys.stdout = Logger('testlog.log')
    main()
    sys.stdout.close()