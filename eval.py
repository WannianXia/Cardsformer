from typing import OrderedDict
from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
from Model.PredictionModel import PredictionModel
from Model.ModelWrapper import Model as PolicyModel

from transformers import AutoModel, AutoTokenizer
import torch
from Algo.encoder import Encoder
import pandas as pd

import logging

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('Cardsformer')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)


name_list = pd.read_csv('Env/classical_cards.csv')['name'].tolist()

NUM_ROUNDS = 1
checkpoint_path = 'your/policy/model/path/here.ckpt'
device_number = '0'


model =PolicyModel(device=device_number)
checkpoint_states = torch.load(checkpoint_path)
model.get_model().load_state_dict(checkpoint_states)

prediction_model = PredictionModel()
checkpoint_states = torch.load("your/prediction/model/path/here.tar", map_location='cuda:'+device_number)['model_state_dict']

# unwrap the prediction model
new_state_dict = OrderedDict()
for k, v in checkpoint_states.items():
    name = k[7:]
    new_state_dict[name] = v

prediction_model.load_state_dict(new_state_dict)
prediction_model.to('cuda:' + str(device_number))
prediction_model.eval()
game = Hearthstone()
device = 'cuda:'+ device_number
env = Environment(game, device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
encoder = Encoder(model=auto_model, tokenizer=tokenizer)
encoder.to(device)

# load graph

position, obs, options, done, episode_return = env.initial()
win = [0, 0]
for i in range(NUM_ROUNDS):
    while True:
        num_options = len(options)

        hand_card_embed = encoder.encode(obs['hand_card_names'])
        minion_embed = encoder.encode(obs['minion_names'])
        weapon_embed = encoder.encode(obs['weapon_names'])
        secret_embed = encoder.encode(obs['secret_names'])
        with torch.no_grad():

            hand_card_embed = encoder.encode(obs['hand_card_names'])
            minion_embed = encoder.encode(obs['minion_names'])
            weapon_embed = encoder.encode(obs['weapon_names'])
            secret_embed = encoder.encode(obs['secret_names'])
            with torch.no_grad():
                next_state = prediction_model([hand_card_embed, minion_embed, weapon_embed, obs['hand_card_scalar_batch'], obs['minion_scalar_batch'], obs['hero_scalar_batch']])
            obs['next_minion_scalar'] = next_state[0]
            obs['next_hero_scalar'] = next_state[1]
            with torch.no_grad():
                agent_output = model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor = True)
            # uncomment the following line to see the action-value results of each available action
            # for i in range(num_options):
            #     log.info('--ACTION-- {} --VALUE-- {}'.format(options[i].FullPrint(), agent_output.reshape(-1)[i]))
            agent_output = agent_output.argmax()
            action_idx = int(agent_output.cpu().detach().numpy())
        action = options[action_idx]
        log.info(action.FullPrint())  # print the performing action
        position, obs, options, done, episode_return, _ = env.step(action)
        log.info(env.Hearthstone.game.FullPrint())  # print current game state
        if done:
            if episode_return > 0:
                win[0] += 1
            elif episode_return < 0:
                win[1] += 1
            else:
                log.info("No winner???")
            break
log.info(win)

