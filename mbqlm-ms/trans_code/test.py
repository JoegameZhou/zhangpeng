import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from pytorch_helper import load_wiki, prepare
from optimization import linear_schedule_with_warmup_iterator
import argparse
import yaml
import torch
# from torch import cat
# from torch import randn
from pytorch_QA_CNN_quantum import QA_quantum as QA_CNN_quantum_model

parser = argparse.ArgumentParser()
parser.add_argument(
	'--config-yml',
	default = 'pytorch_config.yml'
	)
args = parser.parse_args()
args.config_yml = 'pytorch_config.yml'
config = yaml.load(open(args.config_yml),Loader=yaml.FullLoader)

# train, dev, test = load_wiki(config['train']['data'], filter = config['train']['clean']) 
# alphabet, embeddings = prepare([train, test, dev], dim = config['model']['embedding_dim'], is_embedding_needed = True, fresh = True)

# generator = QA_data_generator(config, train, alphabet)

# dataset = GeneratorDataset(generator,column_names=['data','label']).batch(32)

# data_iter = dataset.create_tuple_iterator(num_epochs=1)

# for data, label in data_iter:
#     dat = data
#     lab = label
#     print(dat)
#     print(dat.shape)
#     print(lab.shape)
#     # print(dat.size())
#     # print(data2.size())
#     # print(lab.size())
#     break

# temp = Tensor(np.array([[2,2,2],[3,3,3]]),mindspore.int32)
# a = temp.squeeze(1)

# b = torch.randn(2,3)
# c = b.squeeze(1)
# rand = ops.StandardNormal()
# logits = Tensor(np.array([[0.1,0.1,0.1,0.1,0.1,0.1]]),mindspore.float32)
# labels = Tensor(np.array([[1,0,1,1,1,0]]),mindspore.float32)
# loss = nn.SoftmaxCrossEntropyWithLogits()


# print(loss(logits,labels))

# logits = Tensor([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
# labels = Tensor([0, 3], mindspore.int32)
# sparse_softmax_cross = ops.SparseSoftmaxCrossEntropyWithLogits()
# loss = sparse_softmax_cross(logits, labels)
# print(loss)

# count = 0
# for i in linear_schedule_with_warmup_iter(10,150*136/100, 150*136):
#     if count == 10:
#         break 
#     else:
#         print(i)
#     count += 1

# count = 0
for i in linear_schedule_with_warmup_iterator(0.001,150*136/100, 150*136):
    # if count == 10:
    #     break
    print(i)
    # count += 1