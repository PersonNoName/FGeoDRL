import h5py
import torch.utils.data as data
import torch


class DataFromH5File(data.Dataset):
    def __init__(self, filepath, mode):
        self.state = None
        self.theorem = None
        self.param = None
        self.reward = None
        self.mode = mode
        h5File = h5py.File(filepath, 'r')
        self.state = h5File['state']
        if self.mode == 'SL_theorem':
            self.theorem = h5File['theorem']
        elif self.mode == 'SL_param':
            self.param = h5File['param']
        elif self.mode == 'RL':
            self.theorem = h5File['theorem']
            self.reward = h5File['reward']

    def __getitem__(self, idx):
        state = torch.tensor(self.state[idx], requires_grad=True, dtype=torch.double).cuda()
        if self.mode == 'SL_theorem':
            theorem = torch.tensor(self.theorem[idx], requires_grad=True, dtype=torch.double).cuda()
            return state, theorem
        elif self.mode == 'SL_param':
            param = torch.tensor(self.param[idx], requires_grad=True, dtype=torch.double).cuda()
            return state, param
        elif self.mode == 'RL':
            theorem = torch.tensor(self.theorem[idx], requires_grad=True, dtype=torch.double).cuda()
            reward = torch.tensor(self.reward[idx], requires_grad=True, dtype=torch.double).cuda()
            return state, theorem, reward

    def __len__(self):
        return len(self.state)


class Dataset:
    def __init__(self,
                 mode,
                 save_path):
        self.mode = mode
        self.sava_path = save_path

    def createDataset(self, states, params=None, theorems=None, rewards=None):
        with h5py.File(self.sava_path, 'w') as file:
            state_shape = states[0].shape
            file.create_dataset('state', data=states,
                                maxshape=(None, state_shape[0], state_shape[1], state_shape[2]))
            if self.mode == 'SL_theorem':
                file.create_dataset('theorem', data=theorems, maxshape=(None, 1))
            elif self.mode == 'SL_param':
                param_shape = params[0].shape
                file.create_dataset('param', data=params, maxshape=(None, param_shape[0], param_shape[1]))
            elif self.mode == 'RL':
                file.create_dataset('theorem', data=theorems, maxshape=(None, 1))
                file.create_dataset('reward', data=rewards, maxshape=(None, 1))

    def addDataset(self, state, param=None, theorem=None, reward=None):
        with h5py.File(self.sava_path, 'a') as file:
            dataset_state = file['state']

            dataset_state.resize(dataset_state.shape[0] + state.shape[0], axis=0)
            dataset_state[-state.shape[0]:] = state
            if self.mode == 'SL_theorem':
                if state.shape[0] != theorem.shape[0]:
                    print('添加数据集中的数据个数没有对齐，请检查相关代码')
                    return
                dataset_theorem = file['theorem']
                dataset_theorem.resize(dataset_theorem.shape[0] + theorem.shape[0], axis=0)
                dataset_theorem[-theorem.shape[0]:] = theorem
            if self.mode == 'SL_param':
                if state.shape[0] != len(param):
                    print('添加数据集中的数据个数没有对齐，请检查相关代码')
                    return
                dataset_param = file['param']
                dataset_param.resize(dataset_param.shape[0] + param.shape[0], axis=0)
                dataset_param[-param.shape[0]:] = param
            elif self.mode == 'RL':
                if state.shape[0] != theorem.shape[0] and state.shape[0] != reward.shape[0]:
                    print('添加数据集中的数据个数没有对齐，请检查相关代码')
                    return
                dataset_theorem = file['theorem']
                dataset_reward = file['reward']
                dataset_theorem.resize(dataset_theorem.shape[0] + theorem.shape[0], axis=0)
                dataset_reward.resize(dataset_reward.shape[0] + reward.shape[0], axis=0)
                dataset_theorem[-theorem.shape[0]:] = theorem
                dataset_reward[-reward.shape[0]:] = reward

    # return DataLoader
    def loadDataset(self, batch_size=32, shuffle=True):
        trainset = DataFromH5File(self.sava_path, mode=self.mode)
        dataloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


if __name__ == '__main__':
    import numpy as np
    params = np.zeros((10, 8, 26))
    states = np.zeros((10, 64, 20, 30))
    dataset = Dataset(mode='SL_param', save_path='../Database/test.h5')
    # dataset.createDataset(states=states, params=params)
    # dataset.addDataset(state=states, param=params)
    dataloader = dataset.loadDataset(8)

    for a, b in dataloader:
        print(a.shape, b.shape)
