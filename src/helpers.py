#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_conf_matrix(true, pred, labels=None, title=None, cbarlabel=None, ax=None):
    """
        Plot and show a confusion matrix
        
         Parameters
        -------------------
            true : True Y labels
            pred : Predicted labels
            ax : A matplotlib axis to be plotted, if none, one will be created

         Returns
        -------------------
            None
    """
    
    # Get Confusion Matrix
    cm = confusion_matrix(true, pred)
    
    # Set up axis
    ax = ax if ax else plt.gca()
    im = ax.imshow(cm)
    cbar = ax.figure.colorbar(im)
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    if title:
        ax.set_title(title)
    if cbarlabel:
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')
        
    # Plot values
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    
#     plt.show()

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
    targets = cards['colors'].apply(lambda x: x[0])
    return pd.DataFrame({'path': paths, 'target': targets})

