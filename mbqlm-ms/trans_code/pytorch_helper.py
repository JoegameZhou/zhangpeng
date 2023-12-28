# -*- coding:utf-8-*-
#! /usr/bin/env python3.4
from typing import Tuple
import numpy as np
import random,os,math
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

import sklearn
import multiprocessing
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import evaluation
import string
from nltk import stem
from tqdm import tqdm
# import chardet
import re
from functools import wraps
from numpy.random import seed 

import yaml
import argparse

from mindspore.dataset import GeneratorDataset

seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--config-yml',
    default = 'pytorch_config.yml'
    )
args = parser.parse_args()
args.config_yml = 'pytorch_config.yml'
config = yaml.load(open(args.config_yml),Loader=yaml.FullLoader)

dataset = config['train']['data']
isEnglish = config['data_helper']['isEnglish']
isGlove = config['data_helper']['isGlove']
UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False

def cut(sentence, isEnglish = isEnglish):
    #分词函数
    if isEnglish:
        #句子大写字母换小写，按空格分割，存到tokens中
        tokens = sentence.lower().split()
    else:
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

class Alphabet(dict):
    def __init__(self, start_feature_id = 1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

@log_time_delta
def prepare(cropuses, is_embedding_needed = False, dim = 50, fresh = False):
    vocab_file = 'model/voc'
    
    if os.path.exists(vocab_file) and not fresh:
        print(1)
    else:   
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('[UNKNOW]')  
        alphabet.add('END')
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(), corpus["answer"]]:
                for sentence in tqdm(texts):   
                    count += 1
                    tokens = cut(sentence)
                    for token in set(tokens):
                        alphabet.add(token)
    if is_embedding_needed:
        sub_vec_file = 'embedding/sub_vector'
        if os.path.exists(sub_vec_file) and not fresh:
            print(2)
        else:    
            if isGlove:        
                if dim == 50:
                    file_name = '../data/embedding/glove.6B.50d.txt'
                    embeddings = load_text_vec(alphabet, file_name, embedding_size = dim)
                    sub_embeddings = getSubVectorsFromDict(embeddings, alphabet, dim)
                else:
                    file_name = '/mnt/glove.840B.300d.txt'
                    embeddings = load_text_vec(alphabet, file_name, embedding_size = dim)
                    sub_embeddings = getSubVectorsFromDict(embeddings, alphabet, dim)
            else:
                file_name = "../data/embedding/aquaint+wiki.txt.gz.ndim=50.bin"
                embeddings = KeyedVectors.load_word2vec_format(file_name, binary=True)
                sub_embeddings = getSubVectors(embeddings, alphabet)
        return alphabet, sub_embeddings
    else:
        return alphabet

def one_hot_embedding(alphabet):
    embedding=np.zeros([len(alphabet),len(alphabet)])
    for word in alphabet:
        embedding[alphabet[word]][alphabet[word]]=1
    return embedding

def getSubVectors(vectors,vocab,dim = 50):
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    temp_vec = 0
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]]= vectors.word_vec(word)
        else:
            embedding[vocab[word]]= np.random.uniform(-0.25,+0.25,vectors.syn0.shape[1])  #.tolist()
        temp_vec += embedding[vocab[word]]
    temp_vec /= len(vocab)
    for index,_ in enumerate(embedding):
        embedding[index] -= temp_vec
    return embedding

def load_text_vec(alphabet, word_embedding_filename, embedding_size = 100):
    word_embeddings = {}
    with open(word_embedding_filename) as embedding_file:
        i = 0
        for line in embedding_file:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]         #??????????????????????????
                # print (vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    word_embeddings[word] = items[1:]
    return word_embeddings

def getSubVectorsFromDict(word_embeddings, alphabet, dim = 300):
    file = open('missword', 'w')
    embedding = np.zeros((len(alphabet), dim))
    count = 1
    for word in alphabet:
        if word in word_embeddings:
            count += 1
            embedding[alphabet[word]] = word_embeddings[word]
        else:
            file.write(word + '\n')
            embedding[alphabet[word]] = np.random.uniform(-0.5, +0.5, dim)
    file.close()
    return embedding
@log_time_delta

def get_overlap_dict(data_file, alphabet, question_len = 40, answer_len = 40):
    overlap_dict = dict()
    for question in data_file['question'].unique():
        group = data_file[data_file['question'] == question]
        answers = group['answer']
        for answer in answers:
            question_overlap, answer_overlap = overlap_index(question, answer, question_len, answer_len)
            overlap_dict[(question, answer)] = (question_overlap, answer_overlap)
    return overlap_dict

def overlap_index(question, answer, question_len, answer_len, stopwords = []):
    question_set = set(cut(question))
    answer_set = set(cut(answer))
    question_index = np.zeros(question_len)
    answer_index = np.zeros(answer_len)
    overlap = question_set.intersection(answer_set)
    for i, question_token in enumerate(cut(question)[:question_len]):
        value = 1
        if question_token in overlap:
            value = 2
        question_index[i] = value
    for i, answer_token in enumerate(cut(answer)[:answer_len]):
        value = 1
        if answer_token in overlap:
            value = 2
        answer_index[i] = value
    return question_index, answer_index

def position_index(sentence, length):
    index = np.zeros(length)
    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1, min(raw_len + 1, length + 1))
    return index

def encode_to_split(sentence, alphabet, max_sentence = 40):
    indices = []
    tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    while(len(indices) < max_sentence):
        indices += indices[:(max_sentence - len(indices))]
    return indices[:max_sentence]

def transform(flag):
    if flag == 1:
        return [0,1]
    else:
        return [1,0]

def cleanor(data, filter):
    if filter:
        return removeUnanswerdQuestion(data)
    else:
        return data

def load_trec(dataset = dataset, filter = False):
    data_dir = "../data/" + dataset
    datas = []
    data_file = os.path.join(data_dir,"train.txt")
    train = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
    datas.append(cleanor(train, filter))

    data_file = os.path.join(data_dir,'test.txt')
    test = pd.read_csv(data_file,header = None,sep="\t",names=["qid","aid","question","answer","flag"],quoting =3).fillna('')
    datas.append(cleanor(test, filter))

    data_file = os.path.join(data_dir,"dev.txt")
    dev = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
    datas.append(cleanor(dev, filter))

    return tuple(datas)

def load(dataset = dataset, filter = False):
    data_dir = "../data/" + dataset
    datas = []
    for data_name in ['train-all.txt','test.txt']:
        data_file = os.path.join(data_dir,data_name)
        data = pd.read_csv(data_file,header = None,sep="\t",names=["qid","aid","question","answer","flag"],quoting =3).fillna('')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    data_file = os.path.join(data_dir,"dev.txt")
    dev = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
    datas.append(dev)
    return tuple(datas)

def load_wiki(dataset=dataset, filter=False): 
    data_dir = "../data/" + dataset
    datas = []
    for data_name in ['train.txt', "dev.txt",'test.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)

def load_yahoo(dataset=dataset, filter=False):  
    data_dir = "../data/" + dataset
    datas = []
    for data_name in ['train.txt', "dev1.txt",'test.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)

def removeUnanswerdQuestion(df):
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter =df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi = counter[counter>1].index
    return df[df["question"].isin(questions_have_correct) &  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()

@log_time_delta
def batch_gen_with_single(data_file, alphabet, batch_size = 10, question_len = 33, answer_len = 40, overlap_dict = None):
    pairs=[]
    input_num = 7
    for index, row in data_file.iterrows():
        quetion = encode_to_split(row["question"], alphabet, max_sentence = question_len)
        answer = encode_to_split(row["answer"], alphabet, max_sentence = answer_len)
        if overlap_dict == None:
            question_overlap, answer_overlap = overlap_index(row["question"], row["answer"], question_len, answer_len)
        else:
            question_overlap, answer_overlap = overlap_dict[(row["question"], row["answer"])]
        question_position = position_index(row['question'], question_len)
        answer_position = position_index(row['answer'], answer_len)

        label = transform(row["flag"])

        pairs.append((quetion, answer, question_overlap, answer_overlap, question_position, answer_position, label))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]
        yield [[pair[j] for pair in batch]  for j in range(input_num)]
    batch= pairs[n_batches * batch_size:] + [pairs[n_batches * batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [[pair[i] for pair in batch] for i in range(input_num)]

@log_time_delta
def batch_gen_with_point_wise(df, alphabet, batch_size = 10, overlap_dict = None, q_len = 33, a_len = 40):
    input_num = 7
    pairs = []
    for index, row in df.iterrows():
        question = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        if overlap_dict == None:
            q_overlap,a_overlap = overlap_index(row["question"], row["answer"],q_len,a_len)
        else:
            q_overlap,a_overlap = overlap_dict[(row["question"], row["answer"])]
        q_position = position_index(row['question'],q_len)
        a_position = position_index(row['answer'],a_len)
        label = transform(row["flag"])
        pairs.append((question,answer,label,q_overlap,a_overlap,q_position,a_position))
    n_batches = int(len(pairs)*1.0 / batch_size)
    print('n_batch is {}'.format(n_batches))
    pairs = sklearn.utils.shuffle(pairs)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]
    batch = pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]

    return pairs

# class QA_data_generator:
#     def __init__(self, config, pre_data, alphabet):
#         q_max_sent_length = config['model']['max_len']
#         a_max_sent_length = config['model']['max_len']
#         self.pre_data_list = batch_gen_with_point_wise(pre_data, alphabet, config['train']['batch_size'], overlap_dict = None, q_len = q_max_sent_length, a_len = a_max_sent_length)
#         # print(len(self.pre_data_list))
#         self.data = [[pre_data[i] for pre_data in self.pre_data_list]  for i in range(7)]
#         # print(len(self.data))
#         self.labels = self.data[2]
#         # print(len(self.pre_data_list))
#         self.data.pop(2)
#         print(len(self.data))
#         print(len(self.data[0]))
#         print(len(self.labels))

        
#         # print(len(self.data))    
    
#     def __len__(self):
#         return len(self.labels) 
    
#     def __getitem__(self, idx):
#         return self.data[0][idx],self.data[0][idx], self.labels[idx]