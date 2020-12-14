# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
class HAN_Model(nn.Module):
    def __init__(self,vocab_size,embedding_size,gru_size,class_num,is_pretrain=False,weights=None):
        super(HAN_Model, self).__init__()
        if is_pretrain:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_gru = nn.GRU(input_size=embedding_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.word_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)
        self.word_dense = nn.Linear(2*gru_size,2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.sentence_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)
        self.sentence_dense = nn.Linear(2*gru_size,2*gru_size)
        self.fc = nn.Linear(2*gru_size,class_num)
    def forward(self, x,gpu=False):
        sentence_num = x.shape[1]
        sentence_length = x.shape[2]
        x = x.view([-1,sentence_length])
        x_embedding = self.embedding(x)
        word_outputs, word_hidden = self.word_gru(x_embedding)
        attention_word_outputs = torch.tanh(self.word_dense(word_outputs))
        weights = torch.matmul(attention_word_outputs,self.word_context)
        weights = F.softmax(weights,dim=1)
        x = x.unsqueeze(2)
        if gpu:
            weights = torch.where(x!=0,weights,torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))

        weights = weights/(torch.sum(weights,dim=1).unsqueeze(1)+1e-4)

        sentence_vector = torch.sum(word_outputs*weights,dim=1).view([-1,sentence_num,word_outputs.shape[-1]])
        sentence_outputs, sentence_hidden = self.sentence_gru(sentence_vector)
        attention_sentence_outputs = torch.tanh(self.sentence_dense(sentence_outputs))
        weights = torch.matmul(attention_sentence_outputs,self.sentence_context)
        weights = F.softmax(weights,dim=1)
        x = x.view(-1, sentence_num, x.shape[1])
        x = torch.sum(x, dim=2).unsqueeze(2)
        if gpu:
            weights = torch.where(x!=0,weights,torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights,dim=1).unsqueeze(1)+1e-4)
        document_vector = torch.sum(sentence_outputs*weights,dim=1)
        output = self.fc(document_vector)
        return output

if __name__=="__main__":
    han_model = HAN_Model(vocab_size=30000,embedding_size=200,gru_size=50,class_num=4)
    x = torch.Tensor(np.zeros([64,50,100])).long()
    x[0][0][0:10] = 1
    output = han_model(x)
    print (output.shape)