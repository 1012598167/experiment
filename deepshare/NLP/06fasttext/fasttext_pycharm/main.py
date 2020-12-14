# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import Fasttext
from data import AG_Data
import numpy as np
from tqdm import tqdm
import config as argumentparser
config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)

if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
def get_test_result(data_iter,data_set):
    # 生成测试结果
    model.eval()
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        out = model(data)
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy())
    acc = true_sample_num / data_set.__len__()
    return acc
training_set = AG_Data("/AG/train.csv",min_count=config.min_count,
                       max_length=config.max_length,n_gram=config.n_gram)
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
test_set = AG_Data(data_path="/AG/test.csv",min_count=config.min_count,
                   max_length=config.max_length,n_gram=config.n_gram,word2id=training_set.word2id,
                   uniwords_num=training_set.uniwords_num)
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=0)
model = Fasttext(vocab_size=training_set.uniwords_num+100000,embedding_size=config.embed_size,
                 max_length=config.max_length,label_num=config.label_num)
if config.cuda and torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss = -1
for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for data, label in process_bar:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        label = torch.autograd.Variable(label).squeeze()
        out = model(data)
        loss_now = criterion(out, autograd.Variable(label.long()))
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()
        process_bar.set_postfix(loss=loss_now.data.item())
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
    test_acc = get_test_result(test_iter, test_set)
    print("The test acc is: %.5f" % test_acc)