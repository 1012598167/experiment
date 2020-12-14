#coding:utf-8
from torch.utils import data
import os
import nltk
import numpy as np
import pickle
from collections import Counter
class iwslt_Data(data.DataLoader):
    def __init__(self,source_data_name="train.tags.de-en.de",target_data_name="train.tags.de-en.en",source_vocab_size = 30000, target_vocab_size = 30000):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_data, self.target_data, self.target_data_input = self.load_data()
    def load_data(self):

        raw_source_data = open(self.path+"/iwslt14/"+self.source_data_name,encoding="utf-8").readlines()
        raw_target_data = open(self.path+"/iwslt14/"+self.target_data_name,encoding="utf-8").readlines()
        raw_source_data = [x[0:-1] for x in raw_source_data]
        raw_target_data = [x[0:-1] for x in raw_target_data]
        print (len(raw_target_data))
        print (len(raw_source_data))
        source_data = []
        target_data = []
        for i in range(len(raw_source_data)):
            if raw_target_data[i]!="" and raw_source_data[i]!="" and raw_source_data[i][0]!="<" and raw_target_data[i][0]!="<":
                source_sentence = nltk.word_tokenize(raw_source_data[i],language="german")
                target_sentence = nltk.word_tokenize(raw_target_data[i],language="english")
                if len(source_sentence)<=100 and len(target_sentence)<=100:
                    source_data.append(source_sentence)
                    target_data.append(target_sentence)
        if not os.path.exists(self.path + "/iwslt14/source_word2id"):
            source_word2id = self.get_word2id(source_data,self.source_vocab_size)
            target_word2id = self.get_word2id(target_data,self.target_vocab_size)
            self.source_word2id = source_word2id
            self.target_word2id = target_word2id
            pickle.dump(source_word2id, open(self.path + "/iwslt14/source_word2id", "wb"))
            pickle.dump(target_word2id, open(self.path + "/iwslt14/target_word2id", "wb"))
        else:
            self.source_word2id = pickle.load(open(self.path + "/iwslt14/source_word2id", "rb"))
            self.target_word2id = pickle.load(open(self.path + "/iwslt14/target_word2id", "rb"))
        source_data = self.get_id_datas(source_data,self.source_word2id)
        target_data = self.get_id_datas(target_data,self.target_word2id,is_source=False)


        target_data_input = [[2]+sentence[0:-1] for sentence in target_data]
        source_data = np.array(source_data)
        target_data = np.array(target_data)
        target_data_input = np.array(target_data_input)
        return source_data,target_data,target_data_input
    def get_word2id(self,data,word_num):
        words = []
        for sentence in data:
            for word in sentence:
                words.append(word)
        word_freq = dict(Counter(words).most_common(word_num-4))
        word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
        for word in word_freq:
            word2id[word] = len(word2id)
        return word2id
    def get_id_datas(self,datas,word2id,is_source = True):
        for i, sentence in enumerate(datas):
            for j, word in enumerate(sentence):
                datas[i][j] = word2id.get(word,1)
            if is_source:
                datas[i] = datas[i][0:100] +[0]*(100-len(datas[i]))
                datas[i].reverse()
            else:
                datas[i] = datas[i][0:99]+ [3] + [0] * (99 - len(datas[i]))
        return datas

    def __getitem__(self, idx):
        return self.source_data[idx],self.target_data_input[idx], self.target_data[idx]

    def __len__(self):
        return len(self.source_data)
if __name__=="__main__":
    iwslt_data = iwslt_Data()
    print (iwslt_data.source_data.shape)
    print (iwslt_data.target_data_input.shape)
    print (iwslt_data.target_data.shape)
    print (iwslt_data.source_data[0])
    print (iwslt_data.target_data_input[0])
    print (iwslt_data.target_data[0])