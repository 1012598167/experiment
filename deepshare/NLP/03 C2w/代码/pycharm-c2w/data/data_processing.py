# -*- coding: utf-8 -*-
import  json
import nltk
datas = open("./wiki_00",encoding="utf-8").read().splitlines()
num_words = 0
f_train = open("train.txt","w",encoding="utf-8")
f_valid = open("valid.txt","w",encoding="utf-8")
f_test = open("test.txt","w",encoding="utf-8")
for data in datas:
    data = json.loads(data,strict=False)
    sentences = data["text"]
    sentences = sentences.replace("\n\n",". ")
    sentences = sentences.replace("\n",". ")
    sentences = nltk.sent_tokenize(sentences)
    for sentence in sentences:
        sentence = nltk.word_tokenize(sentence)
        if len(sentence)<10 or len(sentence)>100:
            continue
        num_words+=len(sentence)
        sentence = " ".join(sentence) +"\n"
        if num_words<=1000000:
            f_train.write(sentence)
        elif num_words<=1020000:
            f_valid.write(sentence)
        elif num_words<=1040000:
            f_test.write(sentence)
        else:
            exit()
