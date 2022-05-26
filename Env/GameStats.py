import pandas as pd

class GameStats:

    """
        Provide a dict of card name and corresponding descriptions
    """
    def __init__(self):
        data = pd.read_csv("Env/classical_cards.csv", index_col=0)
        self.card_text_dict = {}
        for card in data.iterrows():
            card = card[1]
            if pd.isna(card['text']):
                self.card_text_dict[card['name']] = None
            else:
                self.card_text_dict[card['name']] = card['text']
