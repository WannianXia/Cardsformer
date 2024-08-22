from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
import random
import numpy as np
from Algo.utils import log

from Algo.encoder import Encoder
from transformers import AutoTokenizer, AutoModel



tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
encoder = Encoder(model=auto_model, tokenizer=tokenizer)
encoder.to('cuda:0')

log.info('Data generation begins.')
for j in range(10):
    game = Hearthstone(random_cards=True)
    env = Environment(game)
    data = {
        'hand_card_names': [],
        'minion_names': [],
        'weapon_names': [],
        'hand_card_scalar': [],
        'minion_scalar': [],
        'hero_scalar': [],
        'next_state_minion_scalar': [],
        'next_state_hero_scalar': [],
    }
    position, obs, options, done, episode_return = env.initial()
    for i in range(1000):
        while True:
            num_options = len(options)
            action_idx = random.randint(0, num_options - 1)
            action = options[action_idx]
            player = position
            old_obs = obs
            position, obs, options, done, _, next_state = env.step(action, player)
            data['hand_card_names'].append(encoder.encode(old_obs['hand_card_names']).detach().cpu().numpy())
            data['minion_names'].append(encoder.encode(old_obs['minion_names']).detach().cpu().numpy())
            data['weapon_names'].append(encoder.encode(old_obs['weapon_names']).detach().cpu().numpy())
            data['hand_card_scalar'].append(old_obs['hand_card_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
            data['minion_scalar'].append(old_obs['minion_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
            data['hero_scalar'].append(old_obs['hero_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
            data['next_state_minion_scalar'].append(next_state['minion_scalar'].detach().cpu().numpy().astype(np.int64))
            data['next_state_hero_scalar'].append(next_state['hero_scalar'].detach().cpu().numpy().astype(np.int64))
            if done:
                break
    np.save('off_line_data' + str(j) + '.npy', data)
    log.info('Round {} completed'.format(j))
