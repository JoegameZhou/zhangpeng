import mindspore
import mindspore.nn as nn
import os
import mindspore.ops.operations as P
import pickle
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import mindspore.dataset.text as text
from mindspore.dataset.text import NormalizeForm
from utils.misc import invert_dict
import mindspore.dataset as ds
import mindspore.numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import json

def invert_dict(d):
    return {v: k for k, v in d.items()}

def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, triples, entity_range = batch
    topic_entity = mindspore.ops.Stack(topic_entity)
    question = {k:mindspore.ops.Concat(axis=0)([q[k] for q in question]) for k in question[0]}
    answer = mindspore.ops.Stack(answer)
    entity_range = mindspore.ops.Stack(entity_range)
    return topic_entity, question, answer, triples, entity_range


class Dataset:
    def __init__(self, fn, bert_name, ent2id, rel2id, add_rev=True, training=False):
        print('Reading questions from {}'.format(fn))
        try:
            if bert_name == "bert-base-uncased":
                self.tokenizer = BertTokenizer.from_pretrained('/sdb/xmh/Pretrained/bert-base-uncased/')
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e

        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)

        data = []
        cnt_bad = 0
        cnt = 0
        for line in open(fn):
            instance = json.loads(line.strip())

            question = self.tokenizer(instance['question'].strip(), max_length=64, padding='max_length',
                                      return_tensors="np")
            question_new = {}
            for key in question:
                question_new[key] = Tensor(question[key])
            tmp = question
            question = question_new
            head = instance['entities']
            ans = [ent2id[a['kb_id']] for a in instance['answers']]
            triples = instance['subgraph']['tuples']
            if len(triples) == 0:
                continue
            sub_ents = set(t[0] for t in triples)
            obj_ents = set(t[2] for t in triples)
            entity_range = sub_ents | obj_ents

            is_bad = False
            if all(e not in entity_range for e in head):
                is_bad = True
            if all(e not in entity_range for e in ans):
                is_bad = True

            if is_bad:
                cnt_bad += 1

            if training and is_bad:  # skip bad examples during training
                continue

            cnt += 1
            entity_range = list(entity_range)

            if add_rev:
                supply_triples = []
                # add self relation
                # for e in entity_range:
                #     supply_triples.append([e, self.rel2id['<self>'], e])
                # add reverse relation
                for s, r, o in triples:
                    rev_r = self.rel2id[self.id2rel[r] + '_rev']
                    supply_triples.append([o, rev_r, s])
                triples += supply_triples

            data.append([head, question, ans, triples, entity_range])

        self.data = []
        for index, item in enumerate(data):
            topic_entity, question, answer, triples, entity_range = data[index]
            topic_entity = self.toOneHot(topic_entity, self.ent2id)
            answer = self.toOneHot(answer, self.ent2id)
            triples = mindspore.Tensor(triples)
            if triples.dim() == 1:
                triples = triples.unsqueeze(0)
            entity_range = self.toOneHot(entity_range, self.ent2id)
            self.data.append([topic_entity, question, answer,triples, entity_range])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def toOneHot(self, indices, ent2id):
        vec_len = len(ent2id)
        # onehot = ops.OneHot()
        zeros = ops.Zeros()
        output = zeros((vec_len,), mindspore.float32)
        output[indices] = Tensor(1.0, mindspore.float32)
        # indices = Tensor(np.array(indices), mindspore.int32)
        # on_value, off_value = Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        return output


def load_data(input_dir, bert_name, batch_size, add_rev=False):
    cache_fn = os.path.join(input_dir, 'cache{}_ms_rev_99.pt'.format('_rev' if add_rev else ''))
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, train_data, dev_data = pickle.load(fp)
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'entities.txt')):
            ent2id[line.strip()] = len(ent2id)
        print(len(ent2id))
        rel2id = {}
        for line in open(os.path.join(input_dir, 'relations.txt')):
            rel2id[line.strip()] = len(rel2id)
        # add self relation and reverse relation
        # rel2id['<self>'] = len(rel2id)
        if add_rev:
            for line in open(os.path.join(input_dir, 'relations.txt')):
                rel2id[line.strip()+'_rev'] = len(rel2id)
        print(len(rel2id))

        train_data = Dataset(os.path.join(input_dir, 'train_simple.json'), bert_name, ent2id, rel2id, add_rev=add_rev, training=True)
        dev_data = Dataset(os.path.join(input_dir, 'dev_simple.json'), bert_name, ent2id, rel2id, add_rev=add_rev)
        # test_data = Dataset(os.path.join(input_dir, 'test_simple.json'), bert_name, ent2id, rel2id, batch_size, add_rev=add_rev)

        with open(cache_fn, 'wb') as fp:
            # pickle.dump((ent2id, rel2id, train_data, dev_data, test_data), fp)
            pickle.dump((ent2id, rel2id, train_data, dev_data), fp)

    return ent2id, rel2id, train_data, dev_data


if __name__ == '__main__':
    input_dir = '/sdb/xmh/Projects/Pytorch/GTA/data/CWQ/'
    bert_name = "bert-base-uncased"
    batch_size = 1
    rev = True
    load_data(input_dir, bert_name, batch_size, rev)