# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import Attention_NMT
from data import iwslt_Data
import numpy as np
from tqdm import tqdm
import config as argumentparser
config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)
from nltk.translate.bleu_score import corpus_bleu
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
def get_dev_loss(data_iter):
    model.eval()
    process_bar = tqdm(data_iter)
    loss = 0
    for source_data, target_data_input, target_data in process_bar:
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            target_data_input = target_data_input.cuda()
            target_data = target_data.cuda()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            target_data_input = torch.autograd.Variable(target_data_input).long()
        target_data = torch.autograd.Variable(target_data).squeeze()
        out = model(source_data, target_data_input)
        loss_now = criterion(out.view(-1, 30000), autograd.Variable(target_data.view(-1).long()))
        weights = target_data.view(-1) != 0
        loss_now = torch.sum((loss_now * weights.float())) / torch.sum(weights.float())
        loss+=loss_now.data.item()
    return loss
def get_test_bleu(data_iter):
    model.eval()
    process_bar = tqdm(data_iter)
    refs = []
    preds = []
    for source_data, target_data_input, target_data in process_bar:
        target_input = torch.Tensor(np.zeros([source_data.shape[0], 1])+2)
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            target_input = target_input.cuda().long()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            target_input = torch.autograd.Variable(target_input).long()
        target_data = target_data.numpy()
        out = model(source_data, target_input,mode="test")
        out = np.array(out).T
        tmp_preds = []
        for i in range(out.shape[0]):
            tmp_preds.append([])
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i][j]!=3:
                    tmp_preds[i].append(out[i][j])
                else:
                    break
        preds += tmp_preds
        tmp_refs = []
        for i in range(target_data.shape[0]):
            tmp_refs.append([])
        for i in range(target_data.shape[0]):
            for j in range(target_data.shape[1]):
                if target_data[i][j]!=3 and target_data[i][j]!=0:
                    tmp_refs[i].append(target_data[i][j])
        tmp_refs = [[x] for x in tmp_refs]
        refs+=tmp_refs
    bleu = corpus_bleu(refs,preds)*100
    with open("./data/result.txt","w") as f:
        for i in range(len(preds)):
            tmp_ref = [target_id2word[id] for id in refs[i][0]]
            tmp_pred = [target_id2word[id] for id in preds[i]]
            f.write("ref: "+" ".join(tmp_ref)+"\n")
            f.write("pred: "+" ".join(tmp_pred)+"\n")
            f.write("\n\n")
    return bleu
training_set = iwslt_Data()
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
valid_set = iwslt_Data(source_data_name="IWSLT14.TED.dev2010.de-en.de",target_data_name="IWSLT14.TED.dev2010.de-en.en")
valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
test_set = iwslt_Data(source_data_name="IWSLT14.TED.tst2012.de-en.de",target_data_name="IWSLT14.TED.tst2012.de-en.en")
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
model = Attention_NMT(source_vocab_size=30000,target_vocab_size=30000,embedding_size=256,
                 source_length=100,target_length=100,lstm_size=256)
if config.cuda and torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss = -1
target_id2word = dict([[x[1],x[0]] for x in training_set.target_word2id.items()])
for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for source_data, target_data_input, target_data in process_bar:
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            target_data_input = target_data_input.cuda()
            target_data = target_data.cuda()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            target_data_input = torch.autograd.Variable(target_data_input).long()
        target_data = torch.autograd.Variable(target_data).squeeze()
        out = model(source_data,target_data_input)

        loss_now = criterion(out.view(-1,30000), autograd.Variable(target_data.view(-1).long()))
        weights = target_data.view(-1)!=0
        loss_now = torch.sum((loss_now*weights.float()))/torch.sum(weights.float())
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()
        process_bar.set_postfix(loss=loss_now.data.item())
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
    test_bleu = get_test_bleu(test_iter)
    print("test bleu is:", test_bleu)
    valid_loss = get_dev_loss(valid_iter)
    print ("valid loss is:",valid_loss)
