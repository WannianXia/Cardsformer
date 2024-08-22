import torch
import torch.nn as nn

# The Prediction Model of Cardsformer
class PredictionModel(nn.Module):
    def __init__(self, card_dim = 63, bert_dim = 768, embed_dim = 256, tf_layer = 4, dim_ff = 512, is_train=False):
        super().__init__()
        self.mpnet_embedding = nn.Linear(bert_dim, embed_dim)
        self.card_dim = card_dim
        self.embed_dim = embed_dim
        self.entity_dim  = self.card_dim + self.embed_dim

        self.hand_card_feat_embed = nn.Linear(23, card_dim)
        self.minion_embeding = nn.Linear(26, card_dim)
        self.hero_embedding = nn.Linear(31, card_dim)

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.entity_dim + 1, nhead=8, dim_feedforward=dim_ff, dropout=0.2)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=tf_layer)

        self.heroes_result = nn.Linear(self.entity_dim + 1, 16)
        self.minions_result = nn.Linear(self.entity_dim + 1, 9)
        self.trans_ln = nn.Linear(27, 16)
        self.is_train = is_train



    def forward(self, x):
        hand_card_embed = x[0]
        minion_embed = x[1]
        weapon_embed = x[2]
        hand_card_scalar = x[3]
        minion_scalar = x[4]
        hero_scalar = x[5]

        hand_card_value = torch.tanh(self.mpnet_embedding(hand_card_embed))
        minion_value = torch.tanh(self.mpnet_embedding(minion_embed))
        weapon_value = torch.tanh(self.mpnet_embedding(weapon_embed))
        if not self.is_train:
            hand_card_value = hand_card_value.repeat(hand_card_scalar.shape[0], 1, 1)
            minion_value = minion_value.repeat(minion_scalar.shape[0], 1, 1)
            weapon_value = weapon_value.repeat(hero_scalar.shape[0], 1, 1)

        hand_card_feat = self.hand_card_feat_embed(hand_card_scalar)
        hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)

        minions_feat = self.minion_embeding(minion_scalar)
        minions_feat = torch.cat((minions_feat, minion_value), dim=-1)

        heroes_feat = self.hero_embedding(hero_scalar)
        heroes_feat = torch.cat((heroes_feat, weapon_value), dim=-1)
        
        entities = torch.cat((hand_card_feat, minions_feat, heroes_feat), dim = -2)
        
            
        pos_embedding = torch.tensor([i / 26 for i in range(27)]).to(entities.device)
        entities = entities.reshape(-1, 27, 319)
        entities = torch.cat((entities, pos_embedding.repeat(entities.shape[0], 1).unsqueeze(-1)), dim=-1)
        # entities = entities.reshape(-1, 27, self.entity_dim + 1)

        out = self.transformer(entities.permute(1, 0, 2)).permute(1, 0, 2)
        out = self.trans_ln(out.permute(0, 2, 1)).permute(0, 2, 1)


        minions = out[:, :14, :]
        heroes = out[:, 14:, :]

        next_minions = self.minions_result(minions)
        next_heroes = self.heroes_result(heroes)

        return [next_minions, next_heroes]

