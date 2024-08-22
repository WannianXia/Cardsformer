import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    def __init__(self, card_dim = 64, lm_dim = 768, embed_dim = 256, dim_ff = 512):
        super().__init__()
        # Embedding Language Model output to a lower dimension
        self.card_dim = card_dim
        self.embed_dim = embed_dim
        self.entity_dim  = self.card_dim + self.embed_dim

        self.lm_embedding = nn.Linear(lm_dim, embed_dim)
        self.secret_embedding  = nn.Linear(lm_dim, self.entity_dim - 1)
        self.pos_embedding = nn.Parameter(torch.tensor([i / 31 for i in range(32)]), requires_grad=False)

        self.hand_card_feat_embed = nn.Linear(23, card_dim - 1)
        self.minion_embeding = nn.Linear(26 + 9, card_dim - 1)
        self.hero_embedding = nn.Linear(31 + 16, card_dim - 1)

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.entity_dim, nhead=8, dim_feedforward=dim_ff, dropout=0.0)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        self.out_ln = nn.Linear(self.entity_dim, 64)
        self.scale_out = nn.Sequential(
            nn.Linear(self.entity_dim, 1),
            nn.Softmax(dim=-2)
            )
        self.fn_ln = nn.Linear(64, 1)



    def forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_scalar, minion_scalar, hero_scalar, next_minion_scalar, next_hero_scalar, num_options, actor = True):
        

        hand_card_value = self.lm_embedding(hand_card_embed)
        minion_value = self.lm_embedding(minion_embed)

        secret_value = self.secret_embedding(secret_embed)
        weapon_value = self.lm_embedding(weapon_embed)

            
        if actor:
            hand_card_value = hand_card_value.repeat(num_options, 1, 1)
            minion_value = minion_value.repeat(num_options, 1, 1)
            secret_value = secret_value.repeat(num_options, 1, 1)
            weapon_value = weapon_value.repeat(num_options, 1, 1)


        
        hand_card_feat = self.hand_card_feat_embed(hand_card_scalar)
        minions_feat = self.minion_embeding(torch.cat((minion_scalar, next_minion_scalar), dim=-1))
        heros_feat = self.hero_embedding(torch.cat((hero_scalar, next_hero_scalar), dim=-1))


        hand_card_feat = torch.cat((hand_card_feat, hand_card_value), dim=-1)
        minions_feat = torch.cat((minions_feat, minion_value), dim=-1)
        heros_feat = torch.cat((heros_feat, weapon_value), dim=-1)
        
        entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_value), dim = -2)
        if not actor:
            entities = entities.reshape(-1, 32, self.entity_dim - 1)

        pos_embed = self.pos_embedding.repeat(entities.shape[0], 1).unsqueeze(-1)
        entities = torch.cat((entities, pos_embed), dim=-1)
        temp_out = self.transformer(entities.permute(1, 0, 2)).permute(1, 0, 2)
        out = self.out_ln(temp_out)
        out_scale = self.scale_out(temp_out)
        out = out * out_scale
        out = torch.sum(out, dim=-2)
        out = self.fn_ln(out).squeeze()

        return out

