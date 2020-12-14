# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
class Attention_NMT(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,embedding_size,
                 source_length,target_length,lstm_size,batch_size = 32):
        super(Attention_NMT,self).__init__()
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_size+2*lstm_size,hidden_size=lstm_size,num_layers=1,
                               batch_first=True)
        self.attention_fc_1 = nn.Linear(3*lstm_size, 3*lstm_size) # 注意力机制全连接层1
        self.attention_fc_2 = nn.Linear(3 * lstm_size, 1) # 注意力机制全连接层2
        self.class_fc_1 = nn.Linear(embedding_size+2*lstm_size+lstm_size, 2*lstm_size) # 分类全连接层1
        self.class_fc_2 = nn.Linear(2*lstm_size, target_vocab_size) # 分类全连接层2

    def attention_forward(self,input_embedding,dec_prev_hidden,enc_output):
        prev_dec_h = dec_prev_hidden[0].squeeze().unsqueeze(1).repeat(1, 100, 1)
        atten_input = torch.cat([enc_output, prev_dec_h], dim=-1)
        attention_weights = self.attention_fc_2(F.relu(self.attention_fc_1(atten_input)))
        attention_weights = F.softmax(attention_weights, dim=1)
        atten_output = torch.sum(attention_weights * enc_output, dim=1).unsqueeze(1)
        dec_lstm_input = torch.cat([input_embedding, atten_output], dim=2)
        dec_output, dec_hidden = self.decoder(dec_lstm_input, dec_prev_hidden)
        return atten_output,dec_output,dec_hidden
    def forward(self, source_data,target_data, mode = "train",is_gpu=True):
        source_data_embedding = self.source_embedding(source_data)
        enc_output, enc_hidden = self.encoder(source_data_embedding)
        self.atten_outputs = Variable(torch.zeros(target_data.shape[0],
                                                  target_data.shape[1],
                                                  enc_output.shape[2]))
        self.dec_outputs = Variable(torch.zeros(target_data.shape[0],
                                                target_data.shape[1],
                                                enc_hidden[0].shape[2]))
        if is_gpu:
            self.atten_outputs = self.atten_outputs.cuda()
            self.dec_outputs = self.dec_outputs.cuda()
        # enc_output: bs*length*(2*lstm_size)
        if mode=="train":
            target_data_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)]
            # dec_prev_hidden[0]: 1*bs*lstm_size, dec_prev_hidden[1]: 1*bs*lstm_size
            # dec_h: bs*lstm_size

            for i in range(100):
                input_embedding = target_data_embedding[:,i,:].unsqueeze(1)
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output)
                self.atten_outputs[:,i] = atten_output.squeeze()
                self.dec_outputs[:,i] = dec_output.squeeze()
                dec_prev_hidden = dec_hidden
            class_input = torch.cat([target_data_embedding,self.atten_outputs,self.dec_outputs],dim=2)
            outs = self.class_fc_2(F.relu(self.class_fc_1(class_input)))
        else:
            input_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)]
            outs = []
            for i in range(100):
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output)

                class_input = torch.cat([input_embedding,atten_output,dec_output],dim=2)
                pred = self.class_fc_2(F.relu(self.class_fc_1(class_input)))
                pred = torch.argmax(pred,dim=-1)
                outs.append(pred.squeeze().cpu().numpy())
                dec_prev_hidden = dec_hidden
                input_embedding = self.target_embedding(pred)
        return outs
if __name__=="__main__":
    attention_nmt = Attention_NMT(source_vocab_size=30000,target_vocab_size=30000,embedding_size=256,
                 source_length=100,target_length=100,lstm_size=256,batch_size=64)
    source_data = torch.Tensor(np.zeros([64,100])).long()
    target_data = torch.Tensor(np.zeros([64,100])).long()
    preds = attention_nmt(source_data,target_data,is_gpu=False)
    print (preds.shape)
    target_data = torch.Tensor(np.zeros([64, 1])).long()
    preds = attention_nmt(source_data, target_data,mode="test",is_gpu=False)
    print(np.array(preds).shape)