import os
import shutil
import requests
import pandas as pd

def gather_artwork(cards, filepath='.'):
    """
        Take in a DataFrame of card info from ScryFall.com to
        request and save cropped card artwork
        
         Parameters
        -------------------
            cards: A Pandas DataFrame of mtg card data
            
         Returns
        -------------------
            None
    """
    
    total = cards.shape[0]
    
    for i, row in cards.iterrows():
        if i % 100 == 0:
            print(f"{i} of {total} gathered.")
        try:
            fp = f"{filepath}/art/{row['set']}/"  # Directory location
            fn = f"/{row['id']}.jpg"  # File name
            link = row['image_uris']['art_crop']  # Link to request
            s = f"({i}/{total}) {row['name']}"
    
        except:
            print(row['name'] + ' Failed.')
            continue
    
        # Skip if this file already exists
        if os.path.isfile(fp+fn): continue
        
        # Make request
        response = requests.get(link, stream=True)
        
        # Save image if request was successful
        if response.status_code == 200:
            # Create destination directory if it doesn't exist
            if not os.path.isdir(fp): os.makedirs(fp)
            # Save image
            with open(fp + fn, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
                
                
if __name__ == '__main__':
    dest = os.path.expanduser('~') + '/data'
    data = 'data/cards.json'
    cards = pd.read_json(data)
    gather_artwork(cards, dest)