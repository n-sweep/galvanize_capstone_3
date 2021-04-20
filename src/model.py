import os
import shutil
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import 
keras = tf.keras

color_key = {'CWUBRG'[i]: (i, c) for i, c in enumerate(['colorless','white','blue','black','red','green'])}


def remove_lands(cards):
    """
        Remove lands from a DataFrame of Scryfall.com card data
        -------------------
            cards : A Pandas DataFrame of mtg card data
         Returns
        -------------------
            A Pandas DataFrame of mtg card data with the lands removed
    """
    return cards[~cards['type_line'].apply(lambda x: 'Land' in x)]

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


if __name__ == '__main__':
    cards = pd.read_json('data/cards.json')
    cards = cards[cards['set'].isin(['m10'])]
    cards = mono_color_cards(cards)
    cards = remove_lands(cards)
    