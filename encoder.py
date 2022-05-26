import torch
import torch.nn.functional as F
from numpy import sqrt as sqrt
import pandas as pd
from Env.GameStats import GameStats


class Encoder:
    def __init__(self, model, tokenizer):
        self.encoder = model
        self.tokenizer = tokenizer
        self.cache = {}
        self.game_stats = GameStats()

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)

    def tokens_to_device(self, tokens):
        tok_device = {}
        for key in tokens:
            tok_device[key] = tokens[key].to(self.device)
        return tok_device


    def encode(self, names):
        txt = []
        for name in names:
            if name is None:
                txt.append(None)
            else:
                description = self.game_stats.card_text_dict[name]
                txt.append(description)

        encoded = []
        for sent in txt:

            if sent in self.cache.keys():
                encoded.append(self.cache[sent])
            elif sent is None:
                encoded.append(torch.zeros((1, 768)).to(self.device))
            else:
                encoded_input = self.tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
                encoded_input = self.tokens_to_device(encoded_input)
                with torch.no_grad():
                    model_output = self.encoder(**encoded_input)
                sent_embed = mean_pooling(model_output, encoded_input['attention_mask'])
                sent_embed = F.normalize(sent_embed, p=2, dim=1)
                encoded.append(sent_embed)
                self.cache[sent] = sent_embed
        if len(encoded) == 0:
            return None
        else:
            return torch.cat(encoded, dim=0)  # n * max_length * 768


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
