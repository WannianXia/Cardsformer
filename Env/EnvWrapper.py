import torch


def obs_to_tensor(obs, device):
    skip_list = ["hand_card_names", "minion_names", "weapon_names", "secret_names"]
    for k in obs:
        if k in skip_list:
            continue
        else:
            obs[k] = torch.from_numpy(obs[k]).float().to(device)
    return obs

class Environment:

    """
    This is the Hearthstone(SabberStone) environment wrapper, it provides an initial function and a step function to process actions.
    """
    def __init__(self, game, device='cpu'):
        """ 
            Initialzie this environment wrapper
        """
        self.Hearthstone = game
        self.device = device
        self.episode_return = None
        
    def initial(self):
        initial_position, initial_obs, initial_options, reward, done = self.Hearthstone.reset()
        initial_obs = obs_to_tensor(initial_obs, self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        
        return initial_position, initial_obs, initial_options, initial_done, self.episode_return
            
    def step(self, action, player = None):
        # action: 当前动作
        # player: 'Player1', 'Player2', None， 如果为None则不计算下一个状态，否则计算当前Player的下一个状态
        position, obs, options, reward, done = self.Hearthstone.step(action)
        if player is None:
            next_state = None
        else:
            if player == position:
                current = True
            else:
                current = False
            next_state = self.Hearthstone.get_next_state(current)
            next_state = obs_to_tensor(next_state, self.device)
        self.episode_return += reward
        episode_return = self.episode_return 
        
        if done:
            position, obs, options, _, _ = self.Hearthstone.reset()
            self.episode_return = torch.zeros(1, 1)
            
        obs = obs_to_tensor(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, options, done, episode_return, next_state
