import numpy as np
from torch.utils.data import Dataset



class PredictionDataset(Dataset):

    def __init__(self, id_list, test=False):
        super(Dataset, self).__init__()
        self.data = {
            'hand_card_names': [],
            'minion_names': [],
            'weapon_names': [],
            'hand_card_scalar': [],
            'minion_scalar': [],
            'hero_scalar': [],
            'next_state_minion_scalar': [],
            'next_state_hero_scalar': [],

        }
        for id in id_list:
            if test:
                data_path = './test_data' + str(id) + '.npy'
            else:
                data_path = './off_line_data' + str(id) + '.npy'
            cur_data = np.load(data_path, allow_pickle=True).item()
            for key in cur_data:
                if key == 'secret_names':
                    continue
                self.data[key] += cur_data[key]

    def __len__(self):
        return len(self.data['hand_card_names'])

    def __getitem__(self, index, id=False):
        return self.data['hand_card_names'][index], self.data['minion_names'][index], self.data['weapon_names'][index], self.data['hand_card_scalar'][index], self.data['minion_scalar'][index], self.data['hero_scalar'][index], self.data['next_state_minion_scalar'][index], self.data['next_state_hero_scalar'][index]