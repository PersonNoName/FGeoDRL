import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=[3, 3, 3], in_channels=3, out_channels=8):
        super(ResNet, self).__init__()
        self.inplanes = 8

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)

    def _make_layer(self, block, in_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != in_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, in_channels * block.expansion, stride),
                nn.BatchNorm2d(in_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, in_channels, stride, downsample))
        self.inplanes = in_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, in_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def resnet(in_channels):
    model = ResNet(block=BasicBlock, layers=[3, 3, 3], in_channels=in_channels)
    return model


class Model(nn.Module):
    def __init__(self,
                 output=4,
                 in_cnel=1):
        super(Model, self).__init__()
        # Policy Network
        self.model = resnet(in_cnel)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.policy_conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1))
        self.policy_bn1 = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(in_features=16*5*8, out_features=output)

    def forward(self, state):
        # policy
        s = self.model(state)

        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p).view(-1, 16*5*8))
        prob = self.policy_fc1(p)
        return prob


class newuralnetwork:
    def __init__(self, input_layers, theorem_size, lr=0.1):
        self.theorem_size = theorem_size
        self.model = Model(output=self.theorem_size, in_cnel=input_layers).cuda().double()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()

    def train(self, data_loader):
        self.model.train()

        loss_record = []
        for idx, (state, theorem) in enumerate(data_loader):
            pred_theorem = self.model(state)
            theorem_probs = torch.zeros((state.shape[0], self.theorem_size), dtype=torch.double, requires_grad=True).cuda()
            for i in range(theorem.shape[0]):
                theorem_probs[i][int(theorem[i])] = 1.0
            probs = F.softmax(pred_theorem, dim=1)
            cross_entropy = self.crossloss(theorem_probs, probs)

            loss = torch.mean(cross_entropy)

            loss.backward()

            self.opt.step()

            loss_record.append(loss)

            if idx % 10 == 0:
                print(idx, ": ", loss)
        return loss_record

    def eval(self, state):
        self.model.eval()

        # state = torch.from_numpy(state).double().cuda().double()

        with torch.no_grad():
            prob = self.model(state)

        return prob

    def save_model(self, filename):
        PATH = './model/sl_theorem/E_' + str(filename) + '.pth'
        torch.save(self.model.state_dict(), PATH)

    def load_model(self, filename):
        PATH = './model/sl_theorem/E_' + str(filename) + '.pth'
        self.model.load_state_dict(torch.load(PATH))

    def testAccuracy(self, data_loader):
        total_num = 0
        correct_num = 0
        theorem_statics = {}
        for idx, (state, theorem) in enumerate(data_loader):
            # pred_theorem = self.eval(state)
            # theorem_probs = torch.zeros((1, self.theorem_size), dtype=torch.double, requires_grad=True).cuda()

            # t = torch.argmax(pred_theorem, dim=1)
            # print(t)
            # if t == int(theorem):
            #     correct_num += 1
            # total_num += 1
            #
            # if (idx+1) % 1000 == 0:
            #     print(correct_num/total_num)
            #     break

            if int(theorem) not in theorem_statics:
                theorem_statics[int(theorem)] = 0
            theorem_statics[int(theorem)] += 1
        return theorem_statics

if __name__ == '__main__':
    from RL.Utils.dataset import Dataset
    input_layer = 65
    theorem_size = 197
    network = newuralnetwork(input_layer, theorem_size)
    #
    SL_theorem_Data = Dataset(mode="SL_theorem", save_path='../../Database/SL/pretrain/theorem/E01.h5')
    # theorem_dataloader = SL_theorem_Data.loadDataset(32)
    #
    # network.train(theorem_dataloader)

    theorem_dataloader = SL_theorem_Data.loadDataset(1)
    theorem_statics = network.testAccuracy(data_loader=theorem_dataloader)
    with open('theorem_statics.json', 'w') as file:
        json.dump(theorem_statics, file)



