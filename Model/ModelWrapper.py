from Model.PolicyModel import PolicyModel
import torch

class Model:
    """
    The wrapper for the Cardsformer policy model. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.model = PolicyModel(card_dim, bert_dim, embed_dim, dim_ff).to(torch.device(device))
        self.device = torch.device(device)

    def forward(self, card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor):

        return self.model(card_embed, minion_embed, secret_embed, weapon_embed, obs["hand_card_scalar_batch"], obs["minion_scalar_batch"], obs["hero_scalar_batch"], obs["next_minion_scalar"], obs["next_hero_scalar"], num_options, actor)

    def share_memory(self):
        self.model.share_memory()
        return

    def eval(self):
        self.model.eval()
        return

    def parameters(self):
        return self.model.parameters()

    def get_model(self):
        return self.model
