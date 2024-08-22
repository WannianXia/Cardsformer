import torch
import typing
import traceback
import numpy as np

import torch 
from torch import multiprocessing as mp
import torch.nn.functional as F

from Env.EnvWrapper import Environment
from Env.Hearthstone import Hearthstone
from transformers import AutoModel, AutoTokenizer
from Algo.encoder import Encoder
from Model.PredictionModel import PredictionModel

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

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)

    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)
    return optimizer


def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['Player1', 'Player2']
    
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),  # 游戏是否结束
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                hand_card_embed=dict(size=(T, 11, 768), dtype=torch.float32),
                minion_embed=dict(size=(T, 14, 768), dtype=torch.float32),
                weapon_embed=dict(size=(T, 2, 768), dtype=torch.float32),
                secret_embed=dict(size=(T, 5, 768), dtype=torch.float32),
                hand_card_scalar=dict(size=(T, 11, 23), dtype=torch.float32),
                minion_scalar=dict(size=(T, 14, 26), dtype=torch.float32),
                hero_scalar=dict(size=(T, 2, 31), dtype=torch.float32),
                next_minion_scalar=dict(size=(T, 14, 9), dtype=torch.float32),
                next_hero_scalar=dict(size=(T, 2, 16), dtype=torch.float32),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['Player1', 'Player2']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)
        
        game = Hearthstone()
        env = Environment(game, device)
        
        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        hand_card_embed_buf = {p: [] for p in positions}
        minion_embed_buf = {p: [] for p in positions}
        weapon_embed_buf = {p: [] for p in positions}
        secret_embed_buf = {p: [] for p in positions}
        hand_card_scalar_buf = {p: [] for p in positions}
        minion_scalar_buf = {p: [] for p in positions}
        hero_scalar_buf = {p: [] for p in positions}
        next_minion_scalar_buf = {p: [] for p in positions}
        next_hero_scalar_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        encoder = Encoder(model=auto_model, tokenizer=tokenizer)
        encoder.to(device)
        position, obs, options, done, episode_return = env.initial()
        prediction_model = PredictionModel()
        checkpoint_states = torch.load("Model/model5294.tar", map_location='cpu')['model_state_dict']
        new_state_dict = typing.OrderedDict()
        for k, v in checkpoint_states.items():
            name = k[7:]
            new_state_dict[name] = v
        
        prediction_model.load_state_dict(new_state_dict)
        prediction_model.to(device)
        prediction_model.eval()

        while True:
        # 进行一局游戏
            while True:
                num_options = len(options)
                if num_options == 1:
                    action = options[0]
                else:
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
                    agent_output = agent_output.argmax()
                    if np.random.rand() < flags.exp_epsilon:
                        _action_idx = torch.randint(len(options), (1, ))[0]
                    else:
                        _action_idx = int(agent_output.cpu().detach().numpy())
                    action = options[_action_idx]
                    hand_card_embed_buf[position].append(hand_card_embed)
                    minion_embed_buf[position].append(minion_embed)
                    weapon_embed_buf[position].append(weapon_embed)
                    secret_embed_buf[position].append(secret_embed)
                    hand_card_scalar_buf[position].append(obs["hand_card_scalar_batch"][_action_idx])
                    minion_scalar_buf[position].append(obs["minion_scalar_batch"][_action_idx])
                    hero_scalar_buf[position].append(obs["hero_scalar_batch"][_action_idx])
                    next_minion_scalar_buf[position].append(obs['next_minion_scalar'][_action_idx])
                    next_hero_scalar_buf[position].append(obs['next_hero_scalar'][_action_idx])
                    # save key info buf here
                    size[position] += 1
                
                position, obs, options, done, episode_return, _ = env.step(action)
                if done:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)
                            episode_return = episode_return if p == 'Player1' else -episode_return
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break
                # 一局游戏结束

            for p in positions:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        # TODO
                        buffers[p]['hand_card_embed'][index][t, ...] = hand_card_embed_buf[p][t]
                        buffers[p]['minion_embed'][index][t, ...] = minion_embed_buf[p][t]
                        buffers[p]['weapon_embed'][index][t, ...] = weapon_embed_buf[p][t]
                        buffers[p]['secret_embed'][index][t, ...] = secret_embed_buf[p][t]
                        buffers[p]['hand_card_scalar'][index][t, ...] =	hand_card_scalar_buf[p][t]
                        buffers[p]['minion_scalar'][index][t, ...] = minion_scalar_buf[p][t]
                        buffers[p]['hero_scalar'][index][t, ...] = hero_scalar_buf[p][t]
                        buffers[p]['next_minion_scalar'][index][t, ...] = next_minion_scalar_buf[p][t]
                        buffers[p]['next_hero_scalar'][index][t, ...] = next_hero_scalar_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    hand_card_embed_buf[p] = hand_card_embed_buf[p][T:]
                    minion_embed_buf[p] = minion_embed_buf[p][T:]
                    weapon_embed_buf[p] = weapon_embed_buf[p][T:]
                    secret_embed_buf[p] = secret_embed_buf[p][T:]
                    hand_card_scalar_buf[p] = hand_card_scalar_buf[p][T:]
                    minion_scalar_buf[p] = minion_scalar_buf[p][T:]
                    hero_scalar_buf[p] = hero_scalar_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    next_minion_scalar_buf[p] = next_minion_scalar_buf[p][T:]
                    next_hero_scalar_buf[p] = next_hero_scalar_buf[p][T:]
                    
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
