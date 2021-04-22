#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_conf_matrix(true, pred, ax=None):
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
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(color_key.keys())
    ax.set_yticklabels(color_key.keys())
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
        
    # Plot values
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.show()

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

