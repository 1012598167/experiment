# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
class C2W(nn.Module):
    def __init__(self, config):
        super(C2W, self).__init__()
        self.char_hidden_size = config.char_hidden_size
        self.word_embed_size = config.word_embed_size
        self.lm_hidden_size = config.lm_hidden_size
        self.character_embedding = nn.Embedding(config.n_chars,config.char_embed_size)
        self.sentence_length = config.max_sentence_length
        self.char_lstm = nn.LSTM(input_size=config.char_embed_size,hidden_size=config.char_hidden_size,
                            bidirectional=True,batch_first=True)
        self.lm_lstm = nn.LSTM(input_size=self.word_embed_size,hidden_size=config.lm_hidden_size,batch_first=True)
        self.fc_1 = nn.Linear(2*config.char_hidden_size,config.word_embed_size)
        self.fc_2 =nn.Linear(config.lm_hidden_size,config.vocab_size)

    def forward(self, x):
        input = self.character_embedding(x)
        char_lstm_result = self.char_lstm(input)
        word_input = torch.cat([char_lstm_result[0][:,0,0:self.char_hidden_size],
                                char_lstm_result[0][:,-1,self.char_hidden_size:]],dim=1)
        word_input = self.fc_1(word_input)
        word_input = word_input.view([-1,self.sentence_length,self.word_embed_size])
        lm_lstm_result = self.lm_lstm(word_input)[0].contiguous()
        lm_lstm_result = lm_lstm_result.view([-1,self.lm_hidden_size])
        out = self.fc_2(lm_lstm_result)
        return out
class config:
    def __init__(self):
        self.n_chars = 64
        self.char_embed_size = 50
        self.max_sentence_length = 8
        self.char_hidden_size = 50
        self.lm_hidden_size = 150
        self.word_embed_size = 50
        config.vocab_size = 1000

if __name__=="__main__":
    config = config()
    c2w = C2W(config)
    test = np.zeros([64,16])
    c2w(test)
