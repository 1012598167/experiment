# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        载入模型
        :param path: 模型路径
        :return: None
        '''
        self.load_state_dict(torch.load(path))

    def save(self, path):
        '''
        保存模型
        :param path: 模型路径
        :return: None
        '''
        torch.save(self.state_dict(), path)

    def forward(self):
        pass


if __name__ == '__main__':
    print('Running the BasicModule.py...')
    model = BasicModule()