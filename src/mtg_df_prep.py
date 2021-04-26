#!/usr/bin/env python3

import pandas as pd

def load_color_data(mono=False, colorless=True, land=True):
    """
        Load in cards dataset and prepare it for color prediction

         Parameters
        -------------------
            mono (bool) : Use only mono-colored cards
            colorless (bool) : Include colorless cards
            land (bool) : Include land cards

         Returns
        -------------------
            A Pandas DataFrame of prepped mtg card data
    """
    cards = pd.read_json('data/cards.json')
    # Drop any cards without colors
    cards = cards[cards['colors'].notna()]
    # Give colorless cards a 'C' value
    cards = mono_color_cards(cards) if mono else cards

    if colorless:
        cards = cards if land else remove_land(cards)
        cards['colors'] = cards['colors'].apply(
            lambda x: ['C'] if len(x) < 1 else x
        )
    else:
        cards = cards[cards['colors'].apply(lambda x: x != [])]

    return cards


def remove_land(cards):
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

