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
from mindspore import dtype as mstype


def load_vocab(path):
    vocab_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            vocab_list.append(line)
    return vocab_list

class Dataset:
    def __init__(self, input_dir, fn, bert_name, ent2id, rel2id, training=False):
        # self.ent2id = ent2id
        # vocab_list = load_vocab('/data/xmh/Pretrained/bert-base-uncased/vocab.txt')
        # vocab_list = load_vocab('/sdb/xmh/Pretrained/bert-base-uncased/vocab.txt')
        # vocab = text.Vocab.from_list(vocab_list)

        print('Reading questions from {}'.format(fn))
        try:
            if bert_name == "bert-base-uncased":
                tokenizer = BertTokenizer.from_pretrained('/sdb/xmh/Pretrained/bert-base-uncased/')
                # tokenizer = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100,
                # unknown_token = '[UNK]', lower_case = False, keep_whitespace = False,
                # normalization_form = NormalizeForm.NONE, preserve_unused_token = True,
                # with_offsets = False)
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e


        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        print("fbwq_full/train.txt")
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            sub_map[s].append((p, o))

        data = []
        for line in open(fn):
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 3:
                continue
            if len(line[1]) < 1:
                # print("bad")
                continue
            question = line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            question_2 = question_2[1]
            # question = question_1 + 'NE' + question_2
            question = question_1.strip()
            ans = line[1].split('|')
            hop = int(line[2].strip())

            entity_range = set()
            for p, o in sub_map[head]:
                entity_range.add(o)
                for p2, o2 in sub_map[o]:
                    entity_range.add(o2)
            entity_range = [ent2id[o] for o in entity_range]

            head = [ent2id[head]]
            question = tokenizer(question.strip(), max_length=64, padding='max_length', return_tensors="np")
            question_new = {}
            for key in question:
                question_new[key] = Tensor(question[key])
            tmp = question
            question = question_new
            ans = [ent2id[a] for a in ans]
            data.append([head, question, ans, entity_range])  # hop
            self.question = data

        self.data = []
        for index, item in enumerate(self.question):
            topic_entity, question, answer, entity_range = self.question[index] # hops
            topic_entity = self.toOneHot(topic_entity, ent2id)
            answer = self.toOneHot(answer, ent2id)
            if len(entity_range) > 100:  # 限制数据量
                entity_range = entity_range[:100]
            entity_range = self.toOneHot(entity_range, ent2id)
            self.data.append([topic_entity, question, answer, entity_range])
        print('data number: {}'.format(len(self.data)))

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


def load_data(input_dir, bert_name, batch_size):
    cache_fn = os.path.join(input_dir, 'processed_hop_fixed_ms2-1000.pt')
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples, train_data, test_data = pickle.load(fp)
        triples = mindspore.Tensor(triples)
        print('Train number: {}, test number: {}'.format(len(train_data), len(test_data)))
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/entities.dict')):
            l = line.strip().split('\t')
            ent2id[l[0].strip()] = len(ent2id)

        rel2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/relations.dict')):
            l = line.strip().split('\t')
            rel2id[l[0].strip()] = int(l[1])

        triples = []
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')

            # if l[0].strip() not in ent2id or  l[1].strip() not in rel2id or l[2].strip() not in ent2id:
            #     continue

            s = ent2id[l[0].strip()]
            p = rel2id[l[1].strip()]
            o = ent2id[l[2].strip()]
            triples.append((s, p, o))
            p_rev = rel2id[l[1].strip()+'_reverse']
            triples.append((o, p_rev, s))
        triples = mindspore.Tensor(triples)
        # input_dir, fn, bert_name, ent2id, rel2id, questions, training = False):
        train_data = Dataset(input_dir, os.path.join(input_dir, 'qa_train_webqsp_hop.txt'), bert_name, ent2id, rel2id, training=True)
        test_data = Dataset(input_dir, os.path.join(input_dir, 'qa_test_webqsp_fixed_hop.txt'), bert_name, ent2id, rel2id, training=True)
        # test_data = train_data
        # train_data = ds.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        # train_data = train_data.batch(batch_size=batch_size)
        # test_data = Dataset(input_dir, os.path.join(input_dir, 'qa_test_webqsp_fixed_hop.txt'), bert_name, ent2id, rel2id)  # _fixed
        # test_data = test_data.batch(batch_size=batch_size)

        # test_data = ds.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples, train_data, test_data), fp)

    return ent2id, rel2id, triples, train_data, test_data
